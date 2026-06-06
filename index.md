# What Makes a Food.com Recipe Highly Rated?

**Name:** Kobe

This project analyzes recipes and user ratings from Food.com. The main question is:

**Which recipe characteristics are associated with higher average ratings, and can we predict a recipe's average rating from information available before users review it?**

The dataset contains **83,782 recipes** and **731,927 user interactions**. Each recipe includes preparation time, number of steps, number of ingredients, nutrition information, and tags. Each interaction includes a user rating and review. Since ratings of 0 represent missing ratings rather than real 0-star ratings, I treated them as missing values.

## Introduction

I chose this question because recipe ratings are useful to both people choosing recipes and recipe creators trying to understand what makes a recipe appealing. In this project, I focus on `average_rating`, the mean non-missing rating for each recipe.

The most important columns are:

| Column | Description |
| --- | --- |
| `minutes` | Preparation time for the recipe |
| `n_steps` | Number of instructions in the recipe |
| `n_ingredients` | Number of ingredients |
| `nutrition` | A list-like column containing calories, fat, sugar, sodium, protein, saturated fat, and carbohydrates |
| `tags` | Recipe tags, such as diet type, meal type, or preparation method |
| `rating` | User rating from the interactions table |
| `average_rating` | The response variable created by averaging nonzero ratings for each recipe |

## Data Cleaning and Exploratory Data Analysis

To clean the data, I merged the recipe and interaction tables, replaced rating values of 0 with missing values, and computed each recipe's `average_rating`. I also split the `nutrition` column into separate numeric columns: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbohydrates`.

Most recipes use a moderate number of ingredients, though the distribution is right-skewed because some recipes have many ingredients.

<iframe src="assets/ingredients_distribution.html" width="100%" height="520" frameborder="0"></iframe>

When comparing average rating to the number of ingredients, the relationship is not strongly linear. Ratings are generally high across most ingredient counts, which suggests that ingredient count alone probably cannot explain recipe quality.

<iframe src="assets/ingredient_rating.html" width="100%" height="520" frameborder="0"></iframe>

I also looked at average rating across both preparation time and ingredient-count groups. The differences are small, but very long recipes and recipes with many ingredients do not always receive worse ratings. This supports the idea that recipe quality is influenced by several factors rather than one simple feature.

<iframe src="assets/rating_heatmap.html" width="100%" height="560" frameborder="0"></iframe>

## Assessment of Missingness

The main missingness question is whether `average_rating` is missing completely at random. A recipe's `average_rating` is missing when it has no valid nonzero ratings.

I do not believe `average_rating` is NMAR. The missingness is better explained as MAR because it depends on observable recipe-level behavior, especially how many interactions a recipe has. Recipes with fewer interactions have fewer chances to receive a valid rating.

I ran permutation tests comparing recipes with missing and non-missing `average_rating` across several observed columns. The missingness of `average_rating` was strongly associated with variables such as `n_interactions`, `day_since_first_submission`, `n_steps`, `calories`, and `n_ingredients`. This gives evidence against MCAR and supports MAR.

## Hypothesis Testing

I tested whether simpler recipes tend to have higher average ratings than more complex recipes. I defined simple recipes as recipes with at most the median number of steps. The median number of steps was **9**, so:

- Simple recipes: `n_steps <= 9`
- Complex recipes: `n_steps > 9`

**Null hypothesis:** Simple and complex recipes have the same distribution of `average_rating`. Any observed difference is due to random chance.

**Alternative hypothesis:** Simple recipes tend to have higher `average_rating` than complex recipes.

**Test statistic:**  
`mean average_rating of simple recipes - mean average_rating of complex recipes`

I used a significance level of **0.05**. The observed statistic was about **0.0031**, and the permutation-test p-value was about **0.255**. Since the p-value is greater than 0.05, I failed to reject the null hypothesis. There is not strong evidence that simpler recipes, measured by number of steps, receive higher ratings.

## Framing a Prediction Problem

The prediction task is to predict a recipe's `average_rating`. This is a **regression** problem because the response variable is numerical.

I used RMSE as the main evaluation metric because it measures prediction error in rating units and penalizes larger mistakes more heavily. This is useful because a large rating prediction error is more concerning than a small one.

The model only uses information that would be available before users rate the recipe, such as preparation time, number of steps, ingredients, nutrition information, and tags. I did not use post-publication variables such as `n_interactions` in the prediction model.

## Baseline Model

The baseline model was a linear regression model using four quantitative features:

- `minutes`
- `n_steps`
- `n_ingredients`
- `calories`

These features were imputed with the median and standardized in a single sklearn pipeline. The baseline model had a test RMSE of about **0.64**, a test MAE of about **0.47**, and a test R² that was extremely close to 0 and slightly negative.

This baseline was useful, but it was not very strong. The very low R² means the model explains almost none of the variation in average rating beyond predicting something close to the mean. Recipe ratings are highly concentrated near high values and are also subjective, so simple numeric recipe characteristics do not capture much of the variation in user ratings. The baseline predictions are also concentrated near the overall average rating, suggesting that the linear regression model is learning that giving most recipes a similar high predicted rating is close to the best it can do with these limited features.

## Final Model

The final model improved on the baseline by adding more recipe information while keeping the model interpretable. It used a regularized linear regression model, **Ridge regression**, inside a single sklearn pipeline.

The final model kept the baseline features and added two new feature sources:

- additional nutrition features: `total_fat`, `sugar`, `protein`, and `carbohydrates`
- text features from `tags`, transformed with `CountVectorizer`

I used `GridSearchCV` to tune:

| Hyperparameter | Meaning | Values searched |
| --- | --- | --- |
| `tags__max_features` | Maximum number of tag features kept by CountVectorizer | 75, 150 |
| `selector__k` | Number of transformed features selected by SelectKBest | 50, 75, all |
| `model__alpha` | Ridge regularization strength | 0.1, 1, 10 |

The best settings were:

| Hyperparameter | Best value |
| --- | --- |
| `tags__max_features` | 150 |
| `selector__k` | all |
| `model__alpha` | 10 |

The final model had a test RMSE of about **0.63**, a test MAE of about **0.46**, and an R² of about **0.02**. This is a small improvement over the baseline, and the larger `alpha` value helps address overfitting by shrinking coefficients.

However, the low R² is still important. Even after adding nutrition and tag information, the model explains only a small fraction of the variation in average rating. This suggests that average rating is highly subjective and is likely influenced by factors that are not fully available in recipe metadata, such as individual taste, expectations, popularity, reviewer behavior, and whether users follow the recipe exactly.

The prediction plots reinforce this interpretation. Both the baseline linear regression model and the final Ridge regression model produce predictions that are much more compressed than the actual ratings, mostly around the overall average rating. This suggests that, given only the information available on the recipe page, predicting roughly the same rating for most dishes is close to the optimal strategy, rather than evidence that the model found strong recipe-specific signals.

<iframe src="assets/baseline_prediction_ecdf.html" width="100%" height="540" frameborder="0"></iframe>

<iframe src="assets/final_prediction_ecdf.html" width="100%" height="540" frameborder="0"></iframe>

<iframe src="assets/model_performance.html" width="100%" height="500" frameborder="0"></iframe>

## Fairness Analysis

For fairness, I compared whether the final model performs worse for longer recipes than for quick recipes.

- Group X: recipes that take more than 30 minutes
- Group Y: recipes that take 30 minutes or less
- Evaluation metric: RMSE
- Test statistic: `RMSE(longer recipes) - RMSE(quick recipes)`
- Significance level: 0.05

**Null hypothesis:** The final model is fair across these two preparation-time groups. Its RMSE for longer recipes and quick recipes is roughly the same, and any observed difference is due to random chance.

**Alternative hypothesis:** The model performs worse for longer recipes. Its RMSE for longer recipes is higher than its RMSE for quick recipes.

The final model's RMSE was about **0.66** for recipes longer than 30 minutes and about **0.60** for recipes that take 30 minutes or less. The observed test statistic was about **0.058**. After running a permutation test with shuffled group labels, the p-value was **less than 0.001**.

Since this p-value is below 0.05, I rejected the null hypothesis. There is evidence that the final model performs worse for recipes taking more than 30 minutes than it does for quicker recipes.

<iframe src="assets/fairness_rmse.html" width="100%" height="500" frameborder="0"></iframe>
