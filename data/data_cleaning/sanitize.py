def main():
    writer = open("real_sf_data.tsv", "w")
    with open("stockfish_data.tsv", "r") as f:
        for line in f:
            if '\ufeff' not in line:
                writer.write(line)

if __name__ == "__main__":
    main()