from pmlb import fetch_data

# Returns a pandas DataFrame
adult_data = fetch_data('adult')
print(adult_data.describe())