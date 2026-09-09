from pathlib import Path

chunk_size = 10 * 1024 * 1024
for filename, prefix in (("audios_EMIS_balanced144.zip", "emis144"), ("audios_EMIS_aligned48.zip", "emis48")):
    source = Path(__file__).with_name(filename)
    with source.open("rb") as handle:
        index = 0
        while data := handle.read(chunk_size):
            source.with_name(f"{prefix}.part{index:02d}").write_bytes(data)
            index += 1
    print(prefix, index)
