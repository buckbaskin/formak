import argparse


def selector(i):
    if i:
        return 2
    else:
        return 3


def main(header_loc, source_loc):
    header_template = """int selected();"""
    source_template = """int selected() { return %d; }"""

    header_text = header_template
    source_text = source_template % selector(True)

    with open(header_loc, "w") as f:
        f.write(header_text)

    with open(source_loc, "w") as f:
        f.write(source_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="codegen demo")
    parser.add_argument("--header", type=str)
    parser.add_argument("--source", type=str)

    args = parser.parse_args()

    print(args.header)
    print(args.source)

    main(args.header, args.source)
