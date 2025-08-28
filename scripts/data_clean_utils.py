import numpy as np
import pandas as pd


columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "city_name",
                    "order_day_of_week",
                    "order_month"]


# --- helpers (minimal) -------------------------------------------------
def _clean_str(s: pd.Series) -> pd.Series:
    """Trim, normalize, and turn common 'NaN'-like tokens into NA."""
    return (
        s.astype("string", copy=False)
         .str.strip()
         .replace({"": pd.NA, "NaN": pd.NA, "NaN ": pd.NA, "nan": pd.NA, "NA": pd.NA})
    )

def _parse_dt(s: pd.Series) -> pd.Series:
    """Safe mixed-format datetime parse; unparseable -> NaT (pandas >=2.0 supports format='mixed')."""
    return pd.to_datetime(_clean_str(s), format="mixed", errors="coerce")


def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower, axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",

            # accept both typo and correct spellings
            "time_orderd": "order_time",
            "time_ordered": "order_time",
            "time_order_picked": "order_picked_time",
            "time_ordered_picked": "order_picked_time",

            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            # "time_taken(min)": "time_taken",  # keep commented if not needed
        }, axis=1)
    )


def data_cleaning(data: pd.DataFrame):
    # normalize column names inside too (keeps original flow intact)
    df = data.copy()
    df.columns = df.columns.str.strip().str.lower()

    # robust filters for minors and 6-star anomalies
    age_num = pd.to_numeric(df.get('age'), errors='coerce')
    ratings_num = pd.to_numeric(df.get('ratings'), errors='coerce')
    minors_data = df.loc[age_num < 18]
    minor_index = minors_data.index.tolist()
    six_star_data = df.loc[ratings_num == 6]
    six_star_index = six_star_data.index.tolist()

    return (
        df
        .drop(columns="id", errors="ignore")
        .drop(index=minor_index)                                                # Minor riders dropped
        .drop(index=six_star_index)                                             # 6-star ratings dropped
        .assign(
            # city column out of rider id
            city_name = lambda x: _clean_str(x['rider_id']).str.split("RES").str.get(0),

            # numeric casts (safe)
            age = lambda x: pd.to_numeric(x['age'], errors='coerce'),
            ratings = lambda x: pd.to_numeric(x['ratings'], errors='coerce'),

            # absolute values for location-based columns
            restaurant_latitude = lambda x: pd.to_numeric(x['restaurant_latitude'], errors='coerce').abs(),
            restaurant_longitude = lambda x: pd.to_numeric(x['restaurant_longitude'], errors='coerce').abs(),
            delivery_latitude = lambda x: pd.to_numeric(x['delivery_latitude'], errors='coerce').abs(),
            delivery_longitude = lambda x: pd.to_numeric(x['delivery_longitude'], errors='coerce').abs(),

            # order date to datetime and feature extraction
            order_date = lambda x: pd.to_datetime(_clean_str(x['order_date']), dayfirst=True, errors='coerce'),
            order_day = lambda x: x['order_date'].dt.day,
            order_month = lambda x: x['order_date'].dt.month,
            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend = lambda x: x['order_date'].dt.day_name().str.lower().isin(["saturday","sunday"]).astype(int),

            # time-based columns (robust mixed parsing)
            order_time = lambda x: _parse_dt(x['order_time']),
            order_picked_time = lambda x: _parse_dt(x['order_picked_time']),

            # time taken to pick order (use total_seconds to handle midnight/day wrap)
            pickup_time_minutes = lambda x: (
                (x['order_picked_time'] - x['order_time']).dt.total_seconds() / 60
            ),

            # hour in which order was placed
            order_time_hour = lambda x: x['order_time'].dt.hour,

            # time of the day when order was placed
            order_time_of_day = lambda x: x['order_time_hour'].pipe(time_of_day),

            # categorical columns
            weather = lambda x: (
                _clean_str(x['weather'])
                .str.lower()
                .str.replace("conditions", "", regex=False)  # remove with/without trailing space
                .str.strip()
                .replace("nan", np.nan)
            ),
            traffic = lambda x: _clean_str(x['traffic']).str.lower(),
            type_of_order = lambda x: _clean_str(x['type_of_order']).str.lower(),
            type_of_vehicle = lambda x: _clean_str(x['type_of_vehicle']).str.lower(),
            festival = lambda x: _clean_str(x['festival']).str.lower(),
            city_type = lambda x: _clean_str(x['city_type']).str.lower(),

            # multiple_deliveries column
            multiple_deliveries = lambda x: pd.to_numeric(x['multiple_deliveries'], errors='coerce'),
            # target column modifications (leave as-is if not needed now)
            # time_taken = lambda x: (x['time_taken'].astype('string').str.replace("(min)", "", regex=False).str.strip().astype(int))
        )
        .drop(columns=["order_time", "order_picked_time"])
    )


def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']

    return (
        data
        .assign(**{
            col: (
                np.where(pd.to_numeric(data[col], errors='coerce') < threshold, np.nan, data[col].values)
            )
            for col in location_columns if col in data.columns
        })
    )


# extract day, day name, month and year
def extract_datetime_features(ser):
    date_col = pd.to_datetime(ser, dayfirst=True, errors='coerce')

    return (
        pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "year": date_col.dt.year,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
            }
        )
    )


def time_of_day(ser):
    return (
        pd.cut(
            ser, bins=[0, 6, 12, 17, 20, 24], right=True, include_lowest=True,
            labels=["after_midnight", "morning", "afternoon", "evening", "night"]
        )
    )


def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = data.drop(columns=[c for c in columns if c in data.columns], errors="ignore")
    return df


def calculate_haversine_distance(df):
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']

    if not all(col in df.columns for col in location_columns):
        return df

    lat1 = pd.to_numeric(df[location_columns[0]], errors='coerce')
    lon1 = pd.to_numeric(df[location_columns[1]], errors='coerce')
    lat2 = pd.to_numeric(df[location_columns[2]], errors='coerce')
    lon2 = pd.to_numeric(df[location_columns[3]], errors='coerce')

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)


def create_distance_type(data: pd.DataFrame):
    return(
        data.assign(
            distance_type=pd.cut(
                data["distance"], bins=[0, 5, 10, 15, 25], right=False,
                labels=["short", "medium", "long", "very_long"]
            )
        )
    )


def perform_data_cleaning(data: pd.DataFrame):
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=columns_to_drop)
    )
    # keep your original behavior (drop any NA). adjust if too aggressive.
    return cleaned_data.dropna()


if __name__ == "__main__":
    # data path for data
    DATA_PATH = "swiggy.csv"

    # read the data from path
    df = pd.read_csv(DATA_PATH)
    print('swiggy data loaded successfuly')

    perform_data_cleaning(df)
