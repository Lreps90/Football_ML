import pandas as pd


def create_import_file(df, output_file_path, provider="", market_name="", selection_name=""):
    """
    Creates a CSV file with match details formatted for import.

    Args:
        df (pd.DataFrame): DataFrame containing 'home_team' and 'away_team' columns.
        output_file_path (str): Path to save the output CSV file.
        provider (str): Name of the provider. Default is an empty string.
        market_name (str): Name of the market. Default is an empty string.
        selection_name (str): Name of the selection. Default is an empty string.
    """
    try:
        # Create Match column
        match_df = df['home_team'] + ' v ' + df['away_team']

        # Create the output DataFrame
        output_df = pd.DataFrame({
            'EventName': match_df,
            'Provider': provider,
            'MarketName': market_name,
            'SelectionName': selection_name
        })

        # Save to CSV file
        output_df.to_csv(output_file_path, index=False)

        # Print success message
        print(f"File created and saved successfully at: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")