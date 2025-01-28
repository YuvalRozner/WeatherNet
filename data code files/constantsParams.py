BEGINING_OF_YEAR = "01010000"
ENDING_OF_YEAR = "12312350"
START_YEAR = 2005
END_YEAR = 2007

DATA_DIRECTORY = "data/tryNew/"

STATIONS_LIST = {
    "Newe Yaar": "186",
    "Tavor Kadoorie": "13",
    "Yavneel": "11"
}

columns = [
    "Date Time", "BP (hPa)", "DiffR (w/m^2)", "Grad (w/m^2)", "NIP (w/m^2)", "RH (%)",
    "TD (degC)", "TDmax (degC)", "TDmin (degC)", "WD (deg)", "WDmax (deg)",
    "WS (m/s)", "Ws1mm (m/s)", "Ws10mm (m/s)", "WSmax (m/s)", "STDwd (deg)"
]

COLUMN_PAIRS = [
    ("date", "Date Time"),
    ("BP", "BP (hPa)"),
    ("DiffR", "DiffR (w/m^2)"),
    ("Grad", "Grad (w/m^2)"),
    ("NIP", "NIP (w/m^2)"),
    ("RH", "RH (%)"),
    ("TD", "TD (degC)"),
    ("TDmax", "TDmax (degC)"),
    ("TDmin", "TDmin (degC)"),
    ("WD", "WD (deg)"),
    ("WDmax", "WDmax (deg)"),
    ("WS", "WS (m/s)"),
    ("WS1mm", "Ws1mm (m/s)"),
    ("Ws10mm", "Ws10mm (m/s)"),
    ("WSmax", "WSmax (m/s)"),
    ("STDwd", "STDwd (deg)")
]

COLUMNS_TO_REMOVE = ['date_for_sort', 'BP (hPa)', 'Time', 'Grad (w/m^2)', 'DiffR (w/m^2)', 'NIP (w/m^2)', 'Ws10mm (m/s)', 'Ws1mm (m/s)']

VALUES_TO_FILL = ['TD (degC)', 'TDmin (degC)', 'TDmax (degC)', 'RH (%)']

NA_VALUES = ['None', 'null', '-', '', ' ', 'NaN', 'nan', 'NAN']