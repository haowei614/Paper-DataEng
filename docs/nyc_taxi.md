# NYC TLC Yellow Taxi Trip Records — Data Dictionary

Source: NYC Taxi & Limousine Commission (TLC), *Yellow Taxi Trip Records*.
Each row is a single completed yellow-taxi trip. Fields below follow the
official TLC data dictionary for the trip-record parquet files.

| column | type | description |
| --- | --- | --- |
| `VendorID` | int | Code for the provider that supplied the record. `1` = Creative Mobile Technologies, LLC; `2` = VeriFone Inc. |
| `tpep_pickup_datetime` | datetime | Date and time when the meter was engaged (trip start). |
| `tpep_dropoff_datetime` | datetime | Date and time when the meter was disengaged (trip end). Must be at or after `tpep_pickup_datetime`. |
| `passenger_count` | int | Number of passengers in the vehicle, entered by the driver. |
| `trip_distance` | float | Elapsed trip distance in miles reported by the taximeter. Non-negative. |
| `RatecodeID` | int | Final rate code in effect at trip end. `1` = Standard rate; `2` = JFK; `3` = Newark; `4` = Nassau or Westchester; `5` = Negotiated fare; `6` = Group ride. |
| `store_and_fwd_flag` | string | Whether the record was held in vehicle memory before sending to the vendor ("store and forward"). `Y` = store and forward trip; `N` = not a store and forward trip. |
| `PULocationID` | int | TLC Taxi Zone in which the taximeter was engaged (pickup zone). |
| `DOLocationID` | int | TLC Taxi Zone in which the taximeter was disengaged (dropoff zone). |
| `payment_type` | int | How the passenger paid. `1` = Credit card; `2` = Cash; `3` = No charge; `4` = Dispute; `5` = Unknown; `6` = Voided trip. |
| `fare_amount` | float | Time-and-distance fare calculated by the meter. Non-negative. |
| `extra` | float | Miscellaneous extras and surcharges (e.g. the $0.50 and $1.00 rush-hour and overnight charges). |
| `mta_tax` | float | $0.50 MTA tax automatically triggered based on the metered rate in use. |
| `tip_amount` | float | Tip amount. Automatically populated for credit-card tips; cash tips are not included. |
| `tolls_amount` | float | Total amount of all tolls paid in the trip. |
| `improvement_surcharge` | float | $0.30 improvement surcharge assessed on hailed trips at the flag drop. |
| `total_amount` | float | Total amount charged to passengers. Does not include cash tips. Should approximate the sum of the fare and surcharge components. |
| `congestion_surcharge` | float | Total amount collected for the NYS congestion surcharge. |
| `Airport_fee` | float | $1.25 fee for pickups at LaGuardia and John F. Kennedy airports. |

## Integrity expectations

- `tpep_dropoff_datetime` must be strictly after `tpep_pickup_datetime`.
- `passenger_count`, `trip_distance`, and all monetary amounts are non-negative.
- `RatecodeID` ∈ {1,2,3,4,5,6}; `payment_type` ∈ {1,2,3,4,5,6}.
- `store_and_fwd_flag` ∈ {`Y`, `N`}.
- `total_amount` is approximately the sum of `fare_amount`, `extra`, `mta_tax`,
  `tip_amount`, `tolls_amount`, `improvement_surcharge`, `congestion_surcharge`,
  and `Airport_fee`.
