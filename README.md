# victor

Measure the time of a search in the collection of vectors using Milvus.

### Results

Setup:
- Hardware: Google Compute instance, Docker optimized OS, 4-core CPU, 16gb, 80gb ssd.
- Milvus v0.10.5: standard container (docker.io/milvusdb/milvus:0.10.5-cpu-d010621-4eda95)

| Size on disk  | Index type    | Request time avg | Initial warmup|
| :-----------: |:-------------:|:----------------:|:-------------:|
| 15gb          | IVF_FLAT      | 8s               |60s            |
| 9gb           | IVF_SQ8       | 130-400ms        |50s            |

