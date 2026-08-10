# ==============================================================================
# Program    : Multiprocess Image/Data Processor (Challenge Project)
# Objective  : Perform CPU-heavy matrix transformations in parallel across CPU cores.
# Concept    : Multiprocessing CPU Acceleration & GIL Bypass
# Why Used   : ProcessPoolExecutor distributes CPU-bound pixel math across physical CPU cores.
# ==============================================================================

from concurrent.futures import ProcessPoolExecutor
import time

def process_image_chunk(image_data):
    """Simulates CPU-heavy image filter calculations (grayscale/blur matrix math)."""
    image_name, chunk_size = image_data
    # CPU heavy pixel transformation simulation
    pixel_checksum = sum(x * x for x in range(chunk_size))
    return f"{image_name} (Filtered Checksum: {pixel_checksum})"

def main():
    print("==========================================================")
    print("           MULTIPROCESS IMAGE/DATA PROCESSOR              ")
    print("==========================================================")

    # List of high-resolution images to process (image_name, pixel_count)
    image_queue = [
        ("photo_4k_01.jpg", 10_000_000),
        ("photo_4k_02.jpg", 10_000_000),
        ("photo_4k_03.jpg", 10_000_000),
        ("photo_4k_04.jpg", 10_000_000)
    ]

    start_time = time.time()

    # What is used : ProcessPoolExecutor()
    # Why it is used: Spawns separate Python interpreter processes on distinct CPU cores
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image_chunk, image_queue))

    elapsed = time.time() - start_time

    print("\n---------------- IMAGE PROCESSING RESULTS ----------------")
    for res in results:
        print(f"Processed: {res}")
    print(f"Total Parallel Processing Time: {elapsed:.2f} seconds")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    main()
