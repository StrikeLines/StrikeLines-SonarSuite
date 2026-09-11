from dataclasses import dataclass, field
import pyxtf
from pathlib import Path
import numpy as np


@dataclass
class XTFWrapper:
    file_path: Path
    file_header: pyxtf.XTFFileHeader = field(default_factory=pyxtf.XTFFileHeader)
    packets: dict = field(default_factory=dict)
    num_ch: int = 2
    sonar_data: list = field(default_factory=list)

    def __post_init__(self):
        # Navigation and attitude required downstream are embedded in the
        # sonar ping headers, so avoid decoding unrelated XTF packet types.
        self.file_header, self.packets = pyxtf.xtf_read(
            str(self.file_path), types=[pyxtf.XTFHeaderType.sonar]
        )
        self.num_ch = self.file_header.NumberOfSonarChannels
        sonar_packets = self.packets.get(pyxtf.XTFHeaderType.sonar, [])
        if not sonar_packets:
            raise ValueError(f"No sonar ping packets found in {self.file_path}")

        # read pings and construct starboard and portside picture
        for ch in range(self.num_ch):
            try:
                self.sonar_data.append(
                    np.flipud(
                        pyxtf.concatenate_channel(
                            self.packets[pyxtf.XTFHeaderType.sonar],
                            file_header=self.file_header,
                            channel=ch,
                            weighted=False,
                        )
                    )
                )
            except (IndexError, ValueError):
                channel_rows = []
                for ch_idx, xtf_ping in enumerate(sonar_packets):
                    try:
                        channel_rows.append(np.asarray(xtf_ping.data[ch]))
                    except (AttributeError, IndexError):
                        channel_rows.append(None)
                        print(f"missing ping at position {ch_idx}")
                valid_rows = [row for row in channel_rows if row is not None]
                if not valid_rows:
                    raise ValueError(f"No samples found for XTF channel {ch}")
                first_valid = valid_rows[0]
                previous = first_valid
                for row_idx, row in enumerate(channel_rows):
                    if row is None:
                        channel_rows[row_idx] = previous.copy()
                    else:
                        previous = row
                self.sonar_data.append(np.asarray(channel_rows))

        # concatenate_channel pads variable-length pings to the widest row.
        # Build the range axis from that actual output width rather than the
        # first ping's NumSamples (which can be shorter after a range change).
        self.num_sample_per_ping = int(self.sonar_data[0].shape[1])
        declared_ranges = [
            float(channel_header.SlantRange)
            for ping in sonar_packets
            for channel_header in ping.ping_chan_headers[: self.num_ch]
            if np.isfinite(channel_header.SlantRange)
            and float(channel_header.SlantRange) > 0
        ]
        if declared_ranges:
            outer_range_m = max(declared_ranges)
        else:
            first_header = sonar_packets[0].ping_chan_headers[0]
            outer_range_m = (
                float(first_header.SecondsPerPing)
                * float(sonar_packets[0].SoundVelocity)
            )
        self.x_axis_m = np.linspace(
            0.0, outer_range_m, self.num_sample_per_ping
        )
