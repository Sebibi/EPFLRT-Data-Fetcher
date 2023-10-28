# EPFL Racing Team Data Fetcher
This is a simple script that fetches data from the EPFL Racing Team's website and stores it in a local CSV file.


#### Run the following commands in the terminal from the "data_fetcher" directory

1. Create a virtual environment with `python3 -m venv venv`
2. Activate the virtual environment with `source venv/bin/activate`
3. Upgarde the package installer with `pip install --upgrade pip`
4. Install the dependencies with `pip install -r requirements.txt`
5. Launch the app with `streamlit run data_fetcher_app.py`
6. The app should open in your browser (if not, go to `http://localhost:8501`)


#### Get the data on the app
1. Select a start and an end date in order to search for data
2. Click on the **Fetch R2D sessions** button
3. Once the data is fetched, select on of the available sessions
4. Chose the data you want to download
5. Get a first visualization by plotting the data before downloading !
6. Enter a file name and dowload the data by clicking the **Download data as CSV** button