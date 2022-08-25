import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials


def upload_gsheet(data):
    #don't forgot to give permission to your service account mail
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Put Your own Credentials JSON file
    creds = ServiceAccountCredentials.from_json_keyfile_name("client-secret.json", scope)
    client = gspread.authorize(creds)
    spreadSheetName = 'Clustering Data'     # GSheet Name
    sheetName = 'Sayfa1'                    # Table name



    spreadsheet = client.open(spreadSheetName)
    cluster_data = spreadsheet.worksheet(sheetName)
    cluster_data.delete_rows(2, 10000) # Clean exists sheet
    print('Segment Data loaded into google sheet successfully')
    set_with_dataframe(cluster_data, data)