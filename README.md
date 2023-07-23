# Instacart Product Recommendation - Readme

## Introduction

Ordering groceries and essential items online has become a convenient way for users to restock their supplies. Instacart, a popular grocery order, and delivery app, serves over 500 Million products across more than 40,000 stores in the U.S. and Canada. To enhance the user experience, Instacart aims to provide product recommendations based on users' previous orders and preferences.

In this project, we were provided with anonymized transactional data of customer orders from Instacart. The objective is to predict which products will be in a user's next order. The dataset includes a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, we have access to their past order history (prior data), current order (train data), and future order (test data). Our task is to build a machine learning model that can predict which products are likely to be reordered by a user in their next order.

## Real-World / Business Objectives and Constraints

The primary business objective is to provide accurate product recommendations to users, based on their previous order history. The recommendations should enhance the user experience and simplify the process of ordering groceries online. However, there are certain constraints and considerations for this project:

- **Data Anonymization**: The dataset is anonymized to protect user privacy and contains no personal identifying information.

- **Prediction Performance**: The machine learning model must be highly accurate in predicting which products a user is likely to reorder. This will ensure that the recommendations are useful and relevant to the users.

- **Computational Efficiency**: The model should be computationally efficient to handle the large dataset of over 3 million orders. This will ensure that product recommendations can be generated quickly for a smooth user experience.

- **Scalability**: The solution should be scalable to handle increasing numbers of users and orders as Instacart continues to grow.

## Data

The data provided for this project can be broadly divided into three parts:

1. **Prior Data**: This dataset contains the order history of every user, with nearly 3 to 100 past orders per user.

2. **Train Data**: The train data includes the current order data of every user, with only one order per user. This data will be used to label the dependent variable (reordered).

3. **Test Data**: The test data comprises future order data of every user. It does not contain any product information, and we need to predict which products will be reordered in the next order.

The datasets provided are as follows:

- `orders.csv`: Order details placed by any user, including order ID, user ID, evaluation set (prior/train/test), order number, day of the week, hour of the day, and days since the prior order.

- `order_products__prior.csv`: Contains all product details for any prior order, including order ID, product ID, add-to-cart order, and a flag indicating whether the product is reordered (1/0).

- `order_products__train.csv`: Contains all product details for a train order, with similar columns to `order_products__prior.csv`.

- `products.csv`: Details of each product, including product ID, product name, aisle ID, and department ID.

- `aisles.csv`: Details of aisles, including aisle ID and aisle name.

- `department.csv`: Details of departments, including department ID and department name.

## Machine Learning Problem

The problem we are addressing is a product recommendation task. Given a user's order history and product preferences, we need to predict which products they are likely to reorder in their next order. This is a binary classification problem, where the output will be 1 if the product is likely to be reordered and 0 otherwise.

## Approach

To approach this problem, we will follow the following strategy:

1. **Data Preprocessing and Feature Engineering**: We will preprocess the data and engineer relevant features based on users' order history, products, and other relevant information.

2. **Model Building**: We will train a machine learning model on the train data to predict the probability of products being reordered by a user.

3. **Top Product Recommendations**: Using the trained model, we will select the top probable products for recommendations, based on users' past order history.

4. **Model Evaluation**: We will evaluate the model's performance on the test data to assess its predictive accuracy.

5. **Deployment**: The final model will be deployed to generate real-time product recommendations for Instacart users.

## Conclusion

The Instacart Product Recommendation project aims to enhance the user experience by providing accurate and relevant product recommendations based on users' past order history. By building a machine learning model that predicts which products are likely to be reordered, Instacart can simplify the process of ordering groceries online and increase user satisfaction. With efficient handling of large-scale data and a scalable solution, Instacart can cater to the needs of its growing user base.

Please refer to the notebooks and code files for detailed technical implementation and analysis of the project.
