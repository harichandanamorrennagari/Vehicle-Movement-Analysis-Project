# Vehicle-Movement-Analysis-Project
 >> Main objective of this project is to develop an Edge AI-based solution that can analyse vehicle  movement in and out of a college campus using data from cameras capturing vehicle photos and license  plates. The solution should provide insights on vehicle movement patterns, parking occupancy, and  match vehicles to an approved vehicle database.  
 >> This project aims to detect and recognize vehicle license plates from images, generate entry and exit times for each vehicle, calculate the total parking duration, and visualize vehicle counts per hour to identify peak parking times. Additionally, it simulates a parking lot and assigns vehicles to parking spots with random entry times. The project uses OpenCV for image processing, Tesseract for Optical Character Recognition (OCR), and Pandas for data manipulation and analysis.
 Table of Contents
  > Requirements
  > Setup
  > Project Structure
  > Step-by-Step Process
  > Image Processing and OCR
  > Time Generation
  > Data Storage
  > Peak Hour Analysis
  > Parking Lot Simulation
  > Visualization
  > Running the Project

Requirements
  1. Python 3.x
  2. OpenCV
  3. Tesseract-OCR
  4. NumPy
  5. Pandas
  6. Matplotlib
  7. imutils (for image resizing)
     
Setup
1.Install the required Python packages:
pip install numpy pandas matplotlib imutils pytesseract
2.Install Tesseract-OCR:
sudo apt-get install tesseract-ocr
Project Structure
VehicleLicensePlateDetection/
│
├── Images/
│   ├── vehicle1.jpg
│   ├── vehicle2.jpg
│   ├── ...
│   └── vehicle12.jpg
│
├── vehicles.csv
├── parking_lot.csv
├── main.py
└── README.md

Step-by-Step Process
1.Image Processing and OCR
Read and Resize Images: Load each vehicle image and resize it for consistent processing.
image = cv2.imread(f'Images/vehicle{i}.jpg')
image = imutils.resize(image, width=500)
2.Convert to Grayscale and Apply Filters: Convert the image to grayscale and apply bilateral filtering for noise reduction.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
3.Edge Detection: Use the Canny edge detector to highlight the edges in the image.
edged = cv2.Canny(gray, 170, 200)
4.Contour Detection: Find contours in the edged image, sort them, and select the largest quadrilateral contour as the license plate.
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
5.Masking and OCR: Mask the area other than the detected license plate and run Tesseract OCR to recognize the text.
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
text = pytesseract.image_to_string(new_image, config=('-l eng --oem 1 --psm 3'))

Time Generation
1.Random Entry and Exit Times: Generate random entry (In_Time) and exit (Out_Time) times for each vehicle.
intime = f"{hh:02d}:{mm:02d}"
hh_out = hh + r.randint(1, 3)
mm_out = mm + r.randint(-6, 13)
outtime = f"{hh_out:02d}:{mm_out:02d}"
2.Calculate Total Time: Calculate the total parking duration in minutes.
total_time_minutes = total_time_obj.total_seconds() / 60

Data Storage
1.Store Data in CSV: Store the recognized license plate text, entry, exit times, and total parking duration in a CSV file.
df.to_csv(file_path, mode='a', header=False, index=False)

Peak Hour Analysis
1.Load CSV Data: Load the vehicle data from the CSV file.
df = pd.read_csv(file_path)
2.Count Vehicles per Hour: Count the number of vehicles present for each hour of the day.
hourly_counts = {hour: 0 for hour in range(24)}
hourly_vehicles = {hour: [] for hour in range(24)}
3.Identify Peak Hour: Determine the hour with the maximum vehicle count.
peak_hour = max(hourly_counts, key=hourly_counts.get)

Parking Lot Simulation
1.Initialize Parking Lot Matrix: Create a matrix to represent the parking lot and initialize it with zeros.
parking_lot = np.zeros((rows, cols), dtype=object)
2.Assign Vehicles to Parking Spots: Randomly assign vehicles to available parking spots with random entry times.
parking_lot[position] = vehicle_names[i]
3.Save Parking Lot Data: Save the parking lot matrix with vehicle numbers and entry times to a CSV file.
df_combined.to_csv(output_file_path, index=True)

Visualization
Bar Chart for Vehicle Counts per Hour: Create a bar chart to visualize the number of vehicles present per hour.
plt.bar(hours, counts, color='skyblue')
plt.axvline(x=peak_hour, color='r', linestyle='--', label=f'Peak Hour: {peak_hour:02d}:00 - {peak_hour+1:02d}:00')
plt.legend()
plt.show()

Running the Project
1.Run the Main Script: Execute the main script to process images, recognize license plates, generate times, and perform analysis.
python main.py
2.View Results: Check the output CSV files (vehicles.csv and parking_lot.csv) and the generated visualizations.



In detail explanation in the form of text: 

1. **Installation and Setup**:
   - The required Python packages, `pytesseract` and `tesseract-ocr`, are installed for Optical Character Recognition (OCR) to read the text from images.

2. **Import Libraries**:
   - Essential libraries such as `numpy`, `os`, `datetime`, `cv2` (OpenCV), `imutils`, `pytesseract`, `pandas`, and `random` are imported. `cv2_imshow` from `google.colab.patches` is used to display images.

3. **Vehicle Image Processing**:
   - The code iterates over a set of vehicle images stored in a directory.
   - Each image is loaded, resized, and converted to grayscale.
   - A bilateral filter is applied to reduce noise and keep edges sharp.
   - Edge detection is performed using the Canny edge detector.
   - Contours are found in the edge-detected image and sorted by area.
   - The largest quadrilateral contour, which is likely to be the license plate, is identified.
   - A mask is created to isolate the number plate, and this region is extracted from the original image.

4. **OCR and Data Extraction**:
   - Tesseract OCR is configured and run on the extracted number plate image to recognize text.
   - Random in-time and out-time for the vehicle are generated to simulate vehicle entry and exit times.
   - The total parking duration is calculated from the in-time and out-time.

5. **Data Storage**:
   - The extracted vehicle number, in-time, out-time, and total parking duration are stored in a CSV file (`vehicles.csv`).
   - If the CSV file already exists, new data is appended; otherwise, a new file is created.

6. **Vehicle Movement Analysis**:
   - The CSV file is loaded, and vehicle counts per hour are analyzed.
   - Each vehicle's presence during different hours is tracked.
   - The peak hour, with the maximum number of vehicles, is identified and displayed.
   - A bar chart visualizing the number of vehicles per hour is created using `matplotlib`.

7. **Parking Occupancy Simulation**:
   - A matrix representing the parking lot is initialized.
   - Vehicle numbers are randomly assigned to parking spots.
   - Random in-times are generated for the parked vehicles.
   - A DataFrame representing the parking lot with vehicle numbers and in-times is created and saved to a CSV file (`parking_lot.csv`).

In summary, this project captures vehicle images, processes them to identify license plates, uses OCR to read the plates, logs vehicle entry and exit times, and analyzes the data to provide insights into vehicle movement patterns and parking occupancy. The results include visual representations and CSV files for further analysis.



