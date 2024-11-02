import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st


# Function to load the dataset for unusual spending detection
def load_spending_data(file_path):
    return pd.read_excel(file_path)


# Function to check if expense is normal
def is_expense_normal(data, data_column, user_value, user_income):
    points = np.array(list(zip(data['income'], data[data_column])))
    hull = ConvexHull(points)

    # Check if the user input is inside the convex hull
    user_point = [user_income, user_value]
    new_points = np.vstack([points, user_point])
    new_hull = ConvexHull(new_points)

    # If the hull vertices don't change after adding the user point, the spending is normal
    return list(new_hull.vertices) == list(hull.vertices)


# Function for feature 1: Detecting unusual spending
def feature_1():
    st.title("Financial Management: Detect Unusual Spending")
    st.write("This app helps you detect if your spending is normal or unusual across multiple categories.")

    # Load the dataset
    file_path = "Book1.xlsx"
    data = load_spending_data(file_path)

    # User inputs
    user_income = st.number_input("Enter your income:", min_value=0.0, step=1.0)
    user_rent = st.number_input("Enter your rent:", min_value=0.0, step=1.0)
    user_food = st.number_input("Enter your food spending:", min_value=0.0, step=1.0)
    user_transportation = st.number_input("Enter your transportation spending:", min_value=0.0, step=1.0)
    user_bills = st.number_input("Enter your bills spending:", min_value=0.0, step=1.0)
    user_utilities = st.number_input("Enter your utilities spending:", min_value=0.0, step=1.0)

    # Button to trigger the expense checking logic
    if st.button("Check Expense"):
        # Calculate the total expenses
        total_expenses = user_rent + user_food + user_transportation + user_bills + user_utilities

        # Check if the total expenses exceed the income
        if total_expenses > user_income:
            st.warning(f"Your total spending ({total_expenses}) exceeds your income ({user_income})!")

        # Extract relevant columns for convex hull analysis
        columns = ['rent', 'food', 'transportation', 'bills', 'utilities']
        user_inputs = [user_rent, user_food, user_transportation, user_bills, user_utilities]

        # Initialize results list
        results = []

        # Analyze each category and add the result to the list
        for column, user_input, label in zip(columns, user_inputs,
                                             ['Rent', 'Food', 'Transportation', 'Bills', 'Utilities']):
            is_normal = is_expense_normal(data, column, user_input, user_income)

            # Append result (category and whether it's normal or abnormal) to results list
            status = "Normal" if is_normal else "Abnormal"
            results.append([label, status])

        # Convert results into a pandas DataFrame
        results_df = pd.DataFrame(results, columns=["Category", "Status"])

        # Display the DataFrame as a table in Streamlit
        st.write("Here are the results of your spending analysis:")
        st.dataframe(results_df)


# Function for feature 2: Monthly Expense Prediction with Savings Goal
def feature_2():
    st.title('Monthly Expense Prediction with Savings Goal')

    # Function to load the dataset and train the model
    def load_and_train_model(dataframe):
        X = dataframe[['income']]
        y = dataframe[['rent', 'food', 'transportation', 'bills', 'utilities']]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

    # Function to predict expenses and calculate savings based on user income and desired savings
    def predict_expenses_with_savings(user_income, desired_savings, model):
        user_data = np.array([[user_income]])
        predicted_expenses = model.predict(user_data)

        # Calculate individual expenses
        expenses = {
            'Rent': predicted_expenses[0][0],
            'Food': predicted_expenses[0][1],
            'Transportation': predicted_expenses[0][2],
            'Bills': predicted_expenses[0][3],
            'Utilities': predicted_expenses[0][4]
        }

        # Calculate the total of all expenses
        total_expenses = sum(expenses.values())

        # Calculate remaining money after total expenses
        remaining_income = user_income - total_expenses

        # Check if user can save the desired amount, or show what is possible
        if remaining_income >= desired_savings:
            final_savings = f"{desired_savings} + {remaining_income - desired_savings}"
        else:
            final_savings = f"{remaining_income} (only this much left to save)"

        # Prepare a DataFrame for display
        result_df = pd.DataFrame({
            'Category': list(expenses.keys()) + ['Total Expenses', 'Savings'],
            'Amount': list(expenses.values()) + [total_expenses, final_savings]
        })

        return result_df, remaining_income

    # Function to calculate averages for a salary range
    def calculate_averages(df, user_income, range_value=2000):
        lower_bound = user_income - range_value
        upper_bound = user_income + range_value
        filtered_df = df[(df['income'] >= lower_bound) & (df['income'] <= upper_bound)]

        # Calculate the averages of the expenses within this range
        if not filtered_df.empty:
            avg_expenses = filtered_df[['rent', 'food', 'transportation', 'bills', 'utilities']].mean()
            return avg_expenses
        else:
            return None

    # Function to plot a bar chart of average expenses
    def plot_avg_expenses(avg_expenses):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=avg_expenses.index, y=avg_expenses.values, palette="Blues_d")
        plt.title('Average Monthly Expenses')
        plt.xlabel('Expense Category')
        plt.ylabel('Average Amount')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # User input for income prediction
    st.markdown("""
    ## What type of recommendation would you like? 
    Please choose either **General** or **Personalized** recommendations for your income prediction.
    """)

    option = st.selectbox("Choose Recommendation Type:", ('General', 'Personalized'))

    if option == 'General':
        df = pd.read_excel("dataset.xlsx")  # Update this path to your file
        model = load_and_train_model(df)

    elif option == 'Personalized':
        st.subheader("📊 Personalized Recommendations")
        uploaded_file = st.file_uploader("Please upload your own Excel file", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            model = load_and_train_model(df)
        else:
            st.write("Please upload your Excel file to get personalized recommendations.")

    st.markdown("---")

    user_income = st.number_input('💼 Enter your monthly income:', min_value=0, step=100)
    desired_savings = st.number_input('💸 Enter the amount you want to save:', min_value=0, step=100)

    predict_button = st.button('📊 Predict Expenses', help='Click to predict expenses based on your inputs!')

    if predict_button:
        if option == 'Personalized' and uploaded_file is None:
            st.write("⚠️ Please upload your Excel file for personalized recommendations.")
        else:
            result_df, remaining_income = predict_expenses_with_savings(user_income, desired_savings, model)

            if remaining_income < 0:
                st.write(f"⚠️ Warning: Your expenses exceed your income by {-remaining_income}!")
            else:
                st.write(f'**Predicted Expenses for an income of {user_income}:**')
                st.table(result_df)  # Display the result in tabular format

            avg_expenses = calculate_averages(df, user_income)
            if avg_expenses is not None:
                st.write("**Here are your average monthly expenses based on others with similar incomes:**")
                plot_avg_expenses(avg_expenses)  # Display the bar chart
            else:
                st.write("No data available for this income range.")


# Function for feature 3: Stock Investment Advisor
def feature_3():
    st.title('Automated Stock Investment Advisor')

    # Load stock data from Excel file
    file_path = "all_stocks_5yr.xlsx"

    # Function to load stock data from Excel file
    def load_data_from_excel(file_path):
        df = pd.read_excel(file_path)
        return df

    # Function to train the model
    def train_model(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Expected_Growth'] = df['Close'].pct_change().shift(-1)
        df.dropna(inplace=True)

        # Define features and target
        X = df[['Close']]
        y = df['Expected_Growth']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

    # Function to calculate investment allocations
    def calculate_portfolio_investment(stock_options, investment_amount, target_amount, time_period):
        allocations = {}
        for stock in stock_options:
            allocations[stock] = investment_amount * stock_options[stock]
        return allocations

    # Load the stock data
    stock_data = load_data_from_excel(file_path)
    stock_options = {}

    # Display unique stock names for selection
    unique_stocks = stock_data['Name'].unique()
    selected_stocks = st.multiselect("Select Stocks for Investment:", unique_stocks)

    # User input for investment amounts
    investment_amount = st.number_input("Enter the total amount you want to invest:", min_value=0.0, step=100.0)
    target_amount = st.number_input("Enter your target return amount:", min_value=0.0, step=100.0)
    time_period = st.number_input("Enter the time period (in days) to achieve your target:", min_value=1)

    if st.button("Calculate Investment Allocations"):
        if not selected_stocks:
            st.warning("Please select at least one stock.")
        else:
            for stock in selected_stocks:
                growth_rate = stock_data[stock_data['Name'] == stock]['Expected_Growth'].mean()  # Average growth rate
                stock_options[stock] = growth_rate  # Store growth rates

            allocations = calculate_portfolio_investment(stock_options, investment_amount, target_amount, time_period)
            st.write("**Investment Allocations:**")
            for stock, allocation in allocations.items():
                st.write(f"{stock}: ${allocation:.2f}")


# Main function to run the app
def main():
    st.title("Financial Management App")
    feature_choice = st.sidebar.selectbox("Select a Feature", ("Detect Unusual Spending",
                                                               "Monthly Expense Prediction with Savings Goal",
                                                               "Automated Stock Investment Advisor"))

    if feature_choice == "Detect Unusual Spending":
        feature_1()
    elif feature_choice == "Monthly Expense Prediction with Savings Goal":
        feature_2()
    elif feature_choice == "Automated Stock Investment Advisor":
        feature_3()


if __name__ == "__main__":
    main()
