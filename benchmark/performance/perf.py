import re
from collections import defaultdict


def sum_timings(filepath):
    total_timings = defaultdict(float)
    with open(filepath, 'r') as file:
        for line in file:
            match = re.match(r"(\w+): ([\de\+\-\.]+)", line)
            if match:
                function, time = match.groups()
                total_timings[function] += float(time)
    return total_timings


def main():
    filepath = "large_dump.txt"  
    total_timings = sum_timings(filepath)
    
    print("Total execution times:")
    for func, time in sorted(total_timings.items(), key=lambda x: x[0]):
        print(f"{func}: {time:.6f} seconds")

if __name__ == "__main__":
    main()
