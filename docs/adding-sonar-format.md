# Adding a sonar input format

Each input format is an independent reader adapter under
`src/sidescantools/readers/`. A reader advertises a `format_id` and one or more
file extensions, then implements:

```python
def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
    ...
```

The returned `SonarDataset` is the boundary between vendor parsing and the
SonarSuite processing pipeline. Its validation requires:

- samples shaped as `[channel, ping, sample]`;
- at least port and starboard channels;
- port in channel 0, stored outer-range to nadir;
- starboard in channel 1, stored nadir to outer-range;
- one WGS 84 longitude/latitude and metadata value per ping;
- `seconds_per_ping` and `slant_range` shaped as `[channel, ping]`;
- distances and speeds in SI units and attitude values in degrees; and
- unavailable optional numeric metadata represented by `NaN`, except for
  established fields where the processing API uses zero as its missing-value
  sentinel.

Register the adapter in `sidescantools.readers`. File dialogs, directory
navigation, batch GeoTIFF export, command-line discovery, and Windows startup
then recognize its extensions automatically. If the reader needs another
third-party parser, also include that package in the project dependencies and
Windows build specification.

Add reader tests using a small representative source file or mocked vendor
packets. Every adapter should also pass the shared `SonarDataset` validation
before its data reaches processing.
