import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_feature_importance(model, feature_names):
    os.makedirs("images", exist_ok=True)

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_df)

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.savefig("images/feature_importance.png")
    plt.close()

    print("✅ Feature importance saved")