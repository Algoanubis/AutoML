import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from data_processor import load_and_prepare_data, get_column_names
from models_classification import (
    train_logistic_regression, train_svc, train_gaussienbayes,
    train_knn_classifier, train_random_forest_classifier,
    train_xgboost_classifier, train_lightgbm_classifier
)
from models_regression import (
    train_linear_regression, train_svr,
    train_knn_regression, train_random_forest_regression,
    train_xgboost_regression, train_lightgbm_regression
)

class AutoMLApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
      # Main Window Settings
        self.title("AutoML Studio Pro")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- 1. TARGET COLUMN SELECTION (Dropdown) ---
        self.label_hedef = ctk.CTkLabel(self, text="Select Target Column:", font=("Arial", 22, "bold"))
        self.label_hedef.pack(pady=(60, 15))
        
        # Using OptionMenu instead of Entry
        self.combo_hedef = ctk.CTkOptionMenu(self, width=450, height=50, font=("Arial", 16), 
                                             values=["Please select a file first..."])
        self.combo_hedef.pack(pady=10)
        
        # --- 2. FILE SELECTION BUTTON ---
        self.btn_dosya_sec = ctk.CTkButton(self, text="📂 Select Data File (.csv / .xlsx)", 
                                           font=("Arial", 18, "bold"), width=350, height=60,
                                           fg_color="#27ae60", hover_color="#2ecc71", 
                                           command=self.dosya_sec)
        self.btn_dosya_sec.pack(pady=30)
        
        self.label_dosya_yolu = ctk.CTkLabel(self, text="Selected File: None", text_color="gray", font=("Arial", 14))
        self.label_dosya_yolu.pack(pady=10)
        
        # --- 3. MODELLER (Grid Yapısı) ---
        self.frame_modeller = ctk.CTkFrame(self)
        self.frame_modeller.pack(pady=20, padx=40, fill="both", expand=True)
        
        self.frame_modeller.grid_columnconfigure(0, weight=1)
        self.frame_modeller.grid_columnconfigure(1, weight=1)
        self.frame_modeller.grid_columnconfigure(2, weight=1)
        
        # --- SINIFLANDIRMA MODELLERİ (Mavi - 7 Model) ---
        lbl_class = ctk.CTkLabel(self.frame_modeller, text="Classification Models", font=("Arial", 16, "bold"))
        lbl_class.grid(row=0, column=0, columnspan=3, pady=(10, 5))

        self.btn_logistic = ctk.CTkButton(self.frame_modeller, text="Logistic Regression", font=("Arial", 15, "bold"), height=42, command=self.run_logistic)
        self.btn_logistic.grid(row=1, column=0, padx=10, pady=7, sticky="ew")
        
        self.btn_svc = ctk.CTkButton(self.frame_modeller, text="SVC", font=("Arial", 15, "bold"), height=42, command=self.run_svc)
        self.btn_svc.grid(row=1, column=1, padx=10, pady=7, sticky="ew")
        
        self.btn_gnb = ctk.CTkButton(self.frame_modeller, text="Gaussian NB", font=("Arial", 15, "bold"), height=42, command=self.run_gaussienNB)
        self.btn_gnb.grid(row=1, column=2, padx=10, pady=7, sticky="ew")

        self.btn_knn_cls = ctk.CTkButton(self.frame_modeller, text="KNN Classifier", font=("Arial", 15, "bold"), height=42, command=self.run_knn_classifier)
        self.btn_knn_cls.grid(row=2, column=0, padx=10, pady=7, sticky="ew")

        self.btn_rf_cls = ctk.CTkButton(self.frame_modeller, text="Random Forest Classifier", font=("Arial", 15, "bold"), height=42, command=self.run_randomforest_classifier)
        self.btn_rf_cls.grid(row=2, column=1, padx=10, pady=7, sticky="ew")

        self.btn_xgb_cls = ctk.CTkButton(self.frame_modeller, text="XGBoost Classifier", font=("Arial", 15, "bold"), height=42, command=self.run_xgboost_classifier)
        self.btn_xgb_cls.grid(row=2, column=2, padx=10, pady=7, sticky="ew")

        self.btn_lgbm_cls = ctk.CTkButton(self.frame_modeller, text="LightGBM Classifier", font=("Arial", 15, "bold"), height=42, command=self.run_lightgbm_classifier)
        self.btn_lgbm_cls.grid(row=3, column=0, columnspan=3, padx=10, pady=7, sticky="ew")

        # --- REGRESYON MODELLERİ (Turuncu - 6 Model) ---
        lbl_reg = ctk.CTkLabel(self.frame_modeller, text="Regression Models", font=("Arial", 16, "bold"))
        lbl_reg.grid(row=4, column=0, columnspan=3, pady=(18, 5))

        self.btn_linear = ctk.CTkButton(self.frame_modeller, text="Linear Regression", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_linear)
        self.btn_linear.grid(row=5, column=0, padx=10, pady=7, sticky="ew")
        
        self.btn_svr = ctk.CTkButton(self.frame_modeller, text="SVR", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_svr)
        self.btn_svr.grid(row=5, column=1, padx=10, pady=7, sticky="ew")

        self.btn_knn_reg = ctk.CTkButton(self.frame_modeller, text="KNN Regression", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_knn_regression)
        self.btn_knn_reg.grid(row=5, column=2, padx=10, pady=7, sticky="ew")

        self.btn_rf_reg = ctk.CTkButton(self.frame_modeller, text="Random Forest Regression", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_randomforest_regression)
        self.btn_rf_reg.grid(row=6, column=0, padx=10, pady=7, sticky="ew")

        self.btn_xgb_reg = ctk.CTkButton(self.frame_modeller, text="XGBoost Regression", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_xgboost_regression)
        self.btn_xgb_reg.grid(row=6, column=1, padx=10, pady=7, sticky="ew")

        self.btn_lgbm_reg = ctk.CTkButton(self.frame_modeller, text="LightGBM Regression", font=("Arial", 15, "bold"), height=42, fg_color="#d35400", hover_color="#e67e22", command=self.run_lightgbm_regression)
        self.btn_lgbm_reg.grid(row=6, column=2, padx=10, pady=7, sticky="ew")
        self.secilen_dosya_yolu = None
        
    def dosya_sec(self):
        # Opens the standard OS file dialog
        dosya = filedialog.askopenfilename(
            title="Select Data File for Model",
            filetypes=(("Data Files", "*.csv *.xlsx *.xls"), ("All Files", "*.*"))
        )
        
        if dosya:
            self.secilen_dosya_yolu = dosya
            # Showing only the filename to the user, not the full path
            dosya_adi = os.path.basename(dosya)
            self.label_dosya_yolu.configure(text=f"Selected File: {dosya_adi}", text_color="white")

            basarili, kolonlar = get_column_names(dosya)
            if basarili and kolonlar:
                self.combo_hedef.configure(values=kolonlar) # Load columns into the menu
                self.combo_hedef.set(kolonlar[-1]) # The target column is usually at the end, select it automatically
            else:
                messagebox.showerror("Read Error", "Could not read the columns of the file! Please check the format.")
                self.combo_hedef.configure(values=["Error Occurred"])
                self.combo_hedef.set("Error Occurred")

    def run_logistic(self):
        # 1. SECURITY CHECKS: Is a file selected? Is a target column entered?
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
            
        hedef_sutun = self.combo_hedef.get() 
        # Önceki kodda İngilizceye çevirdiğimiz açılır menü uyarılarıyla birebir aynı olmalı ki hata vermesin:
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return

        # 2. DATA PROCESSING: Sending the file to data_processor and getting X and y
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        
        if not basarili_mi:
            # If the process fails, display the error message from the 'sonuc' variable and stop.
            messagebox.showerror("Data Error", sonuc)
            return
            
        # If successful, unpack X and y
        X, y = sonuc
        
        # 3. ENGINE IGNITION: Send X and y to the Logistic Regression engine
        try:
            # Artık skor değil, 'metrikler' sözlüğü geliyor
            metrikler, model, scaler = train_logistic_regression(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Classification"
            
            # Paketleri açıyoruz
            f1 = metrikler["F1"]
            auc = metrikler["AUC"]
            cm = metrikler["CM"]
            
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Logistik Regresyon modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Logistik Regresyon modeli eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_svc(self):
        # 1. SECURITY CHECKS
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        # Açılır menüdeki İngilizce uyarılarla eşleşmesi için burayı da güncelledik:
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        
        # 2. DATA PREPARATION
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
            
        X, y = sonuc

        # 3. ENGINE IGNITION
        # 3. MOTORU ATEŞLEME (Sınıflandırma Örneği)
        try:
            # Artık skor değil, 'metrikler' sözlüğü geliyor
            metrikler, model, scaler = train_svc(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Classification"
            
            # Paketleri açıyoruz
            f1 = metrikler["F1"]
            auc = metrikler["AUC"]
            cm = metrikler["CM"]
            
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! SVC modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"SVC modeli eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_linear(self):
        # 1. SECURITY CHECKS
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        # Açılır menüdeki İngilizce uyarılarla eşleşmesi için burayı da güncelledik:
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        
        # 2. DATA PREPARATION
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
            
        X, y = sonuc

        # 3. ENGINE IGNITION
        try: 
            metrikler, model, scaler = train_linear_regression(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Linear Regression"

            r2 = metrikler["R2"]
            mae = metrikler["MAE"]
            mse = metrikler["MSE"]
            
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"

            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Lineer Regresyon modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Lineer Regresyon modeli eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_svr(self):
        # 1. SECURITY CHECKS
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        
        # 2. DATA PREPARATION
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
            
        X, y = sonuc

        # 3. ENGINE IGNITION
        try: 
            metrikler, model, scaler = train_svr(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Linear Regression"

            r2 = metrikler["R2"]
            mae = metrikler["MAE"]
            mse = metrikler["MSE"]
            
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"

            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! SVR modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"SVR modeli eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_gaussienNB(self):
        # 1. SECURITY CHECKS
                if not self.secilen_dosya_yolu:
                    messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                    return
                
                hedef_sutun = self.combo_hedef.get() 
                if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
                    messagebox.showerror("Missing Information", "Please select a valid target column!")
                    return
                
                # 2. DATA PREPARATION
                basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
                if not basarili_mi:
                    messagebox.showerror("Data Error", sonuc)
                    return
                    
                X, y = sonuc
        
                # 3. ENGINE IGNITION
                try: 
                    metrikler, model, scaler = train_gaussienbayes(X,y)
                    
                    self.aktif_model = model
                    self.aktif_scaler = scaler
                    self.aktif_model_tipi = "GaussienNB"
        
                    f1 = metrikler["F1"]
                    auc = metrikler["AUC"]
                    cm = metrikler["CM"]
                    
                    mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
                                
                    if f1 >= 0.80:
                        messagebox.showinfo("Model Başarılı!", f"Harika! GaussienNB modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
                    else:
                        messagebox.showwarning("Düşük Başarı", f"GaussienNB modeli başarıyla eğitildi ancak skor düşük.\n\n{mesaj}")
                                                           
                except Exception as e:
                    messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_knn_classifier(self):
            # 1. SECURITY CHECKS
                if not self.secilen_dosya_yolu:
                    messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                    return
                
                hedef_sutun = self.combo_hedef.get() 
                if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
                    messagebox.showerror("Missing Information", "Please select a valid target column!")
                    return
                
                # 2. DATA PREPARATION
                basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
                if not basarili_mi:
                    messagebox.showerror("Data Error", sonuc)
                    return
                    
                X, y = sonuc
        
                # 3. ENGINE IGNITION
                try: 
                    metrikler, model, scaler = train_knn_classifier(X,y)
                    
                    self.aktif_model = model
                    self.aktif_scaler = scaler
                    self.aktif_model_tipi = "KNN Classifier"
        
                    f1 = metrikler["F1"]
                    auc = metrikler["AUC"]
                    cm = metrikler["CM"]
                    
                    mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
                                
                    if f1 >= 0.80:
                        messagebox.showinfo("Model Başarılı!", f"Harika! KNN Classifier modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
                    else:
                        messagebox.showwarning("Düşük Başarı", f"KNN Classifier modeli başarıyla eğitildi ancak skor düşük.\n\n{mesaj}")
                                                           
                except Exception as e:
                    messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")      
                                        
    def run_randomforest_classifier(self):
                # 1. SECURITY CHECKS
                    if not self.secilen_dosya_yolu:
                        messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                        return
                    
                    hedef_sutun = self.combo_hedef.get() 
                    if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
                        messagebox.showerror("Missing Information", "Please select a valid target column!")
                        return
                    
                    # 2. DATA PREPARATION
                    basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
                    if not basarili_mi:
                        messagebox.showerror("Data Error", sonuc)
                        return
                        
                    X, y = sonuc
            
                    # 3. ENGINE IGNITION
                    try: 
                        metrikler, model, scaler = train_random_forest_classifier(X,y)
                        
                        self.aktif_model = model
                        self.aktif_scaler = scaler
                        self.aktif_model_tipi = "Random Forest Classifier"
            
                        f1 = metrikler["F1"]
                        auc = metrikler["AUC"]
                        cm = metrikler["CM"]
                        
                        mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
                                    
                        if f1 >= 0.80:
                            messagebox.showinfo("Model Başarılı!", f"Harika! Random Forest Classifier modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
                        else:
                            messagebox.showwarning("Düşük Başarı", f"Random Forest Classifier modeli başarıyla eğitildi ancak skor düşük.\n\n{mesaj}")
                                                               
                    except Exception as e:
                        messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")      
               
    def run_knn_regression(self):
                # 1. SECURITY CHECKS
                    if not self.secilen_dosya_yolu:
                        messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                        return
                    
                    hedef_sutun = self.combo_hedef.get() 
                    if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
                        messagebox.showerror("Missing Information", "Please select a valid target column!")
                        return
                    
                    # 2. DATA PREPARATION
                    basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
                    if not basarili_mi:
                        messagebox.showerror("Data Error", sonuc)
                        return
                        
                    X, y = sonuc
            
                    # 3. ENGINE IGNITION
                    try: 
                        metrikler, model, scaler = train_knn_regression(X,y)
                        
                        self.aktif_model = model
                        self.aktif_scaler = scaler
                        self.aktif_model_tipi = "KNN Regression"
            
                        r2 = metrikler["R2"]
                        mae = metrikler["MAE"]
                        mse = metrikler["MSE"]
                        
                        mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"
                                    
                        if r2 >= 0.80:
                            messagebox.showinfo("Model Başarılı!", f"Harika! KNN Regression modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
                        else:
                            messagebox.showwarning("Düşük Başarı", f"KNN Regression modeli başarıyla eğitildi ancak skor düşük.\n\n{mesaj}")
                                                               
                    except Exception as e:
                        messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")      
                                               
    def run_randomforest_regression(self):
                    # 1. SECURITY CHECKS
                        if not self.secilen_dosya_yolu:
                            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                            return
                        
                        hedef_sutun = self.combo_hedef.get() 
                        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
                            messagebox.showerror("Missing Information", "Please select a valid target column!")
                            return
                        
                        # 2. DATA PREPARATION
                        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
                        if not basarili_mi:
                            messagebox.showerror("Data Error", sonuc)
                            return
                            
                        X, y = sonuc
                
                        # 3. ENGINE IGNITION
                        try: 
                            metrikler, model, scaler = train_random_forest_regression(X,y)
                            
                            self.aktif_model = model
                            self.aktif_scaler = scaler
                            self.aktif_model_tipi = "Random Forest Regression"
                
                            r2 = metrikler["R2"]
                            mae = metrikler["MAE"]
                            mse = metrikler["MSE"]
                            
                            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"
                                        
                            if r2 >= 0.80:
                                messagebox.showinfo("Model Başarılı!", f"Harika! Random Forest Regression modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
                            else:
                                messagebox.showwarning("Düşük Başarı", f"Random Forest Regression modeli başarıyla eğitildi ancak skor düşük.\n\n{mesaj}")
                                                                   
                        except Exception as e:
                            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")      

    def run_xgboost_classifier(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        hedef_sutun = self.combo_hedef.get()
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
        X, y = sonuc
        try:
            metrikler, model, scaler = train_xgboost_classifier(X, y)
            self.aktif_model, self.aktif_scaler, self.aktif_model_tipi = model, scaler, "XGBoost Classifier"
            f1, auc, cm = metrikler["F1"], metrikler["AUC"], metrikler["CM"]
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! XGBoost Classifier modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"XGBoost Classifier modeli eğitildi ancak skor düşük.\n\n{mesaj}")
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_lightgbm_classifier(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        hedef_sutun = self.combo_hedef.get()
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
        X, y = sonuc
        try:
            metrikler, model, scaler = train_lightgbm_classifier(X, y)
            self.aktif_model, self.aktif_scaler, self.aktif_model_tipi = model, scaler, "LightGBM Classifier"
            f1, auc, cm = metrikler["F1"], metrikler["AUC"], metrikler["CM"]
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! LightGBM Classifier modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"LightGBM Classifier modeli eğitildi ancak skor düşük.\n\n{mesaj}")
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_xgboost_regression(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        hedef_sutun = self.combo_hedef.get()
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
        X, y = sonuc
        try:
            metrikler, model, scaler = train_xgboost_regression(X, y)
            self.aktif_model, self.aktif_scaler, self.aktif_model_tipi = model, scaler, "XGBoost Regression"
            r2, mae, mse = metrikler["R2"], metrikler["MAE"], metrikler["MSE"]
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"
            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! XGBoost Regression modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"XGBoost Regression modeli eğitildi ancak skor düşük.\n\n{mesaj}")
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_lightgbm_regression(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        hedef_sutun = self.combo_hedef.get()
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
        X, y = sonuc
        try:
            metrikler, model, scaler = train_lightgbm_regression(X, y)
            self.aktif_model, self.aktif_scaler, self.aktif_model_tipi = model, scaler, "LightGBM Regression"
            r2, mae, mse = metrikler["R2"], metrikler["MAE"], metrikler["MSE"]
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"
            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! LightGBM Regression modeli yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"LightGBM Regression modeli eğitildi ancak skor düşük.\n\n{mesaj}")
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")
               

if __name__ == "__main__":
    app = AutoMLApp()
    app.mainloop()