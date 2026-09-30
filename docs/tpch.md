# TPC-H — Data Dictionary (subset: customer, orders, lineitem)

Source: TPC Benchmark™ H (TPC-H) specification. Column descriptions below are
drawn from the TPC-H logical schema. This experiment uses scale factor 0.1
generated via DuckDB's `tpch` extension.

Key relationships:

- `customer.c_custkey` is the primary key of `customer`.
- `orders.o_orderkey` is the primary key of `orders`.
- `orders.o_custkey` is a foreign key referencing `customer.c_custkey`.
- `lineitem.l_orderkey` is a foreign key referencing `orders.o_orderkey`;
  together with `l_linenumber` it forms the primary key of `lineitem`.

## customer

| column | type | description |
| --- | --- | --- |
| `c_custkey` | int | Primary key. Unique customer identifier. |
| `c_name` | string | Customer name (e.g. `Customer#000000001`). |
| `c_address` | string | Customer address. |
| `c_nationkey` | int | Foreign key to the nation table (not loaded here); range 0–24. |
| `c_phone` | string | Phone number in the TPC-H format. |
| `c_acctbal` | float | Account balance. May be negative. |
| `c_mktsegment` | string | Market segment. One of `AUTOMOBILE`, `BUILDING`, `FURNITURE`, `HOUSEHOLD`, `MACHINERY`. |
| `c_comment` | string | Free-text comment. |

## orders

| column | type | description |
| --- | --- | --- |
| `o_orderkey` | int | Primary key. Unique order identifier. |
| `o_custkey` | int | Foreign key to `customer.c_custkey`. |
| `o_orderstatus` | string | Order status. One of `O` (open), `F` (fulfilled), `P` (partial). |
| `o_totalprice` | float | Total price of the order. Non-negative. |
| `o_orderdate` | date | Date the order was placed. |
| `o_orderpriority` | string | Priority code, e.g. `1-URGENT`, `2-HIGH`, `3-MEDIUM`, `4-NOT SPECIFIED`, `5-LOW`. |
| `o_clerk` | string | Clerk identifier (e.g. `Clerk#000000001`). |
| `o_shippriority` | int | Shipping priority (integer, typically 0). |
| `o_comment` | string | Free-text comment. |

## lineitem

| column | type | description |
| --- | --- | --- |
| `l_orderkey` | int | Foreign key to `orders.o_orderkey`. Part of the primary key. |
| `l_partkey` | int | Foreign key to the part table (not loaded here). |
| `l_suppkey` | int | Foreign key to the supplier table (not loaded here). |
| `l_linenumber` | int | Line number within the order. Part of the primary key. |
| `l_quantity` | float | Quantity ordered. Positive; typically 1–50. |
| `l_extendedprice` | float | Extended price for the line (quantity × part price). Non-negative. |
| `l_discount` | float | Discount fraction for the line. Between 0.00 and 0.10 inclusive. |
| `l_tax` | float | Tax fraction for the line. Between 0.00 and 0.08 inclusive. |
| `l_returnflag` | string | Return flag. One of `R` (returned), `A` (accepted), `N` (none). |
| `l_linestatus` | string | Line status. One of `O` (open), `F` (finished). |
| `l_shipdate` | date | Date the line item shipped. |
| `l_commitdate` | date | Committed delivery date. |
| `l_receiptdate` | date | Date the line item was received. On or after `l_shipdate`. |
| `l_shipinstruct` | string | Shipping instruction, e.g. `DELIVER IN PERSON`, `COLLECT COD`, `NONE`, `TAKE BACK RETURN`. |
| `l_shipmode` | string | Ship mode, e.g. `AIR`, `RAIL`, `SHIP`, `TRUCK`, `MAIL`, `FOB`, `REG AIR`. |
| `l_comment` | string | Free-text comment. |

## Integrity expectations

- Primary keys (`c_custkey`, `o_orderkey`, and `(l_orderkey, l_linenumber)`) are unique.
- Every `o_custkey` matches an existing `c_custkey`; every `l_orderkey` matches an existing `o_orderkey`.
- `l_discount` ∈ [0.00, 0.10]; `l_tax` ∈ [0.00, 0.08]; `l_quantity` > 0.
- `o_orderstatus` ∈ {`O`, `F`, `P`}; `l_returnflag` ∈ {`R`, `A`, `N`}; `l_linestatus` ∈ {`O`, `F`}.
- `l_receiptdate` ≥ `l_shipdate`.
