import tkinter as tk
from tkinter import filedialog, ttk
from ttkthemes import ThemedTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from PIL import Image, ImageTk
import os

# สร้างหน้าต่าง tkinter
root = ThemedTk(theme="arc")
root.title("Ensemble Occupation Predictor")
root.geometry("1080x720")
root.configure(bg="grey")

data = None
original_categories = None
ensemble_model = None

# ฟังก์ชันเพื่อเลือกไฟล์ CSV และโชว์ข้อมูลใน Treeview
def browse_file():
    global data, original_categories
    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    original_categories = data['Occupation'].astype('category').cat.categories
    show_data(data)

# ฟังก์ชันเพื่อโชว์ข้อมูลใน Treeview
def show_data(data):
    treeview.delete(*treeview.get_children())
    for index, row in data.iterrows():
        values = (row['Age'], row['Sleep Duration'], row['Quality of Sleep'], row['Physical Activity Level'], row['Stress Level'], row['Occupation'])
        treeview.insert("", "end", values=values)
    for col in columns:
        treeview.heading(col, text=col, anchor='center')
        treeview.column(col, anchor='center')


# ฟังก์ชันทำนายและแสดงผลลัพธ์
def train_all_models():
    global original_categories, data, ensemble_model
    try:
        if ensemble_model is None:
            original_categories = data['Occupation'].astype('category').cat.categories
            data['Occupation'] = data['Occupation'].astype('category')
            data['Occupation'] = data['Occupation'].cat.codes
            X = data[['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level']]
            y = data['Occupation']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
            
            logistic_regression_model = LogisticRegression()
            naive_bayes_model = GaussianNB()
            svm_model = SVC()
            decision_tree_model = DecisionTreeClassifier()
            random_forest_model = RandomForestClassifier()
            mlp_model = MLPClassifier()
            adaboost_model = AdaBoostClassifier()
            
            models = [('logistic_regression', logistic_regression_model),
                      ('naive_bayes', naive_bayes_model),
                      ('svm', svm_model),
                      ('decision_tree', decision_tree_model),
                      ('random_forest', random_forest_model),
                      ('mlp', mlp_model),
                      ('adaboost', adaboost_model)]
            
            ensemble_model = VotingClassifier(estimators=models, voting='hard')
            ensemble_model.fit(X_train, y_train)
        else:
            pass
        
        new_data = {
            'Age': int(entry_Age.get()),
            'Sleep Duration': float(entry_Sleep_Duration.get()),
            'Quality of Sleep': int(entry_Quality_of_Sleep.get()),
            'Physical Activity Level': int(entry_Physical_Activity_Level.get()),
            'Stress Level': int(entry_Stress_Level.get())
        }
        new_data_df = pd.DataFrame([new_data])
        
        ensemble_prediction = ensemble_model.predict(new_data_df)
        
        predicted_category = pd.Categorical.from_codes(ensemble_prediction.astype(int), categories=original_categories)
        result_var.set(f'Predicted Occupation: {predicted_category[0]}')
    except Exception as e:
        result_var.showerror("Error", f"An error occurred while predicting: {str(e)}")


def reset_inputs():
        new_data = {
            'Age': entry_Age.delete(0, tk.END),
            'Sleep Duration': entry_Sleep_Duration.delete(0, tk.END),
            'Quality of Sleep': entry_Quality_of_Sleep.delete(0, tk.END),
            'Physical Activity Level':  entry_Physical_Activity_Level.delete(0, tk.END),
            'Stress Level': entry_Stress_Level.delete(0, tk.END),
            "Predict Occupation" : result_var.set("")
        }


# สร้าง Frame สำหรับ Input
frm_input = tk.Frame(root, padx=10, pady=10, bg="grey")
frm_input.grid(row=0, column=0, sticky="nsew")

# สร้าง Canvas สำหรับ Treeview
canvas = tk.Canvas(root, height=600, width=400)
canvas.place(x=300, y=65)

columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Occupation']
treeview = ttk.Treeview(canvas, columns=columns, show='headings')

font_size = 15
style = ttk.Style()
style.configure("Treeview.Heading", font=(None, font_size))

for col in columns:
    treeview.heading(col, text=col)

treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# สร้าง Frame สำหรับ Input ของรูปภาพ
frm_image_input = tk.Frame(root, padx=10, pady=10, bg="#003c5c")
frm_image_input.grid(row=1, column=0, columnspan=2, sticky="nsew")

# สร้าง Label และ Entry สำหรับ Age
label_Age = tk.Label(root, text="Age:", fg="black")
label_Age.grid(row=2, column=0, padx=5, pady=5, sticky="e")

entry_Age = tk.Entry(root)
entry_Age.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# สร้าง Label และ Entry สำหรับ Sleep Duration
label_Sleep_Duration = tk.Label(root, text="Sleep Duration:", fg="black")
label_Sleep_Duration.grid(row=3, column=0, padx=5, pady=5, sticky="e")

entry_Sleep_Duration = tk.Entry(root)
entry_Sleep_Duration.grid(row=3, column=1, padx=5, pady=5, sticky="w")

# สร้าง Label และ Entry สำหรับ Quality of Sleep
label_Quality_of_Sleep = tk.Label(root, text="Quality of Sleep:", fg="black")
label_Quality_of_Sleep.grid(row=4, column=0, padx=5, pady=5, sticky="e")

entry_Quality_of_Sleep = tk.Entry(root)
entry_Quality_of_Sleep.grid(row=4, column=1, padx=5, pady=5, sticky="w")

# สร้าง Label และ Entry สำหรับ Physical Activity Level
label_Physical_Activity_Level = tk.Label(root, text="Physical Activity Level:", fg="black")
label_Physical_Activity_Level.grid(row=5, column=0, padx=5, pady=5, sticky="e")

entry_Physical_Activity_Level = tk.Entry(root)
entry_Physical_Activity_Level.grid(row=5, column=1, padx=5, pady=5, sticky="w")

# สร้าง Label และ Entry สำหรับ Stress Level
label_Stress_Level = tk.Label(root, text="Stress Level:", fg="black")
label_Stress_Level.grid(row=6, column=0, padx=5, pady=5, sticky="e")

entry_Stress_Level = tk.Entry(root)
entry_Stress_Level.grid(row=6, column=1, padx=5, pady=5, sticky="w")

# สร้างปุ่ม Browse CSV
btn_browse = tk.Button(frm_input, text="Open CSV", command=browse_file, bg="green", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_browse.grid(row=11, column=0, columnspan=1, pady=5)

# สร้างปุ่ม Predict Occupation
btn_train_all_models = tk.Button(root, text="Predict Occupation", command=train_all_models, bg="blue", fg="white")
btn_train_all_models.grid(row=8, column=0, columnspan=1, pady=5)

# สร้างปุ่ม Reset Inputs
btn_reset = tk.Button(root, text="Reset Inputs", command=reset_inputs, bg="red", fg="white", relief=tk.FLAT, font=("Arial", 12))
btn_reset.grid(row=9, column=0, columnspan=1, pady=5)

# Label แสดงผลลัพธ์
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.place(x=140, y=225)


root.mainloop()
