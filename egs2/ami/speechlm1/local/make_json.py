import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate JSON metadata file from wav.scp")
    parser.add_argument("--input", "-i", required=True,
                        help="Input wav.scp file (or any file with first-column IDs)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file")

    args = parser.parse_args()

    examples = []
    with open(args.input, "r") as f:
        for line in f:
            col1 = line.strip().split()[0]
            examples.append(col1)

    data = {
        "task": "codec_ssl_sd_event_sad_od_dur30_skip10",
        "vocabularies": [
            "dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/test/token_lists/codec_ssl_token_list",
            "dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/test/token_lists/diar_tokenizer_token_list"
        ],
        "data_files": [
            "dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/test/index_files/wav.scp,codec_ssl,kaldi_ark",
            "dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/test/index_files/diar_tokens_event_sad_od_dur30_skip10,diar_tokenizer,diar_tokens_event_sad_od_dur30_skip10"
        ],
        "num_examples": len(examples),
        "examples": examples
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved JSON to {args.output}, num_examples={len(examples)}")


if __name__ == "__main__":
    main()
