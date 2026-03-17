import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.cluster import KMeans

# Function to perform clustering
def run_clustering():
    try:
        data_input = entry_data.get()
        k = int(entry_k.get())

        # Convert input into list of points
        points = []
        for pair in data_input.split(";"):
            x, y = map(float, pair.split(","))
            points.append([x, y])

        X = np.array(points)

        # Apply KMeans
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(X)

        labels = model.labels_
        centroids = model.cluster_centers_

        # Display result
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Clusters:\n")

        for i, point in enumerate(points):
            result_text.insert(tk.END, f"{point} → Cluster {labels[i]+1}\n")

        result_text.insert(tk.END, "\nCentroids:\n")
        for i, c in enumerate(centroids):
            result_text.insert(tk.END, f"C{i+1}: {c}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# UI Window
root = tk.Tk()
root.title("K-Means Clustering UI")

# Input for data
tk.Label(root, text="Enter points (x,y; x,y; ...):").pack()
entry_data = tk.Entry(root, width=50)
entry_data.pack()

# Input for K
tk.Label(root, text="Enter number of clusters (K):").pack()
entry_k = tk.Entry(root)
entry_k.pack()

# Button
tk.Button(root, text="Run Clustering", command=run_clustering).pack()

# Output box
result_text = tk.Text(root, height=15, width=50)
result_text.pack()

root.mainloop()
