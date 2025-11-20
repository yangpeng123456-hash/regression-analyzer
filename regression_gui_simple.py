import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os

class SimpleRegressionAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("回归分析工具 - 简化版")
        self.root.geometry("600x400")
        
        self.df = None
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="回归分析工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 文件选择
        file_frame = ttk.LabelFrame(main_frame, text="数据文件", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="浏览", command=self.browse_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="加载数据", command=self.load_data).pack(side=tk.LEFT)
        
        # 状态显示
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, pady=(10, 0))
        
        # 分析按钮
        ttk.Button(main_frame, text="执行分析", command=self.analyze_data).pack(pady=20)
        
        # 结果显示
        self.result_text = tk.Text(main_frame, height=15, width=70)
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv")]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def load_data(self):
        if not self.file_path.get():
            messagebox.showerror("错误", "请先选择文件")
            return
        
        try:
            if self.file_path.get().endswith('.csv'):
                self.df = pd.read_csv(self.file_path.get())
            else:
                self.df = pd.read_excel(self.file_path.get())
            
            self.status_var.set(f"成功加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    
    def analyze_data(self):
        if self.df is None:
            messagebox.showerror("错误", "请先加载数据")
            return
        
        try:
            self.status_var.set("正在分析数据...")
            
            # 简单的数据分析
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            result = "数据分析结果:\n"
            result += "=" * 50 + "\n"
            result += f"数据形状: {self.df.shape}\n"
            result += f"数值列: {', '.join(numeric_cols)}\n\n"
            
            if len(numeric_cols) > 0:
                result += "描述性统计:\n"
                result += str(self.df[numeric_cols].describe().round(2)) + "\n\n"
                
                result += "相关性矩阵:\n"
                correlation = self.df[numeric_cols].corr().round(3)
                result += str(correlation) + "\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result)
            self.status_var.set("分析完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_var.set("分析失败")

def main():
    root = tk.Tk()
    app = SimpleRegressionAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
