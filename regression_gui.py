import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac


class RegressionAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("回归分析工具")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # 设置图标（如果有的话）
        try:
            self.root.iconbitmap("analysis_icon.ico")
        except:
            pass

        self.df = None
        self.results = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(main_frame, text="回归分析工具",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="数据文件", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=70).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=1)
        ttk.Button(file_frame, text="加载数据", command=self.load_data).grid(row=0, column=2, padx=(10, 0))

        # 数据显示区域
        data_frame = ttk.LabelFrame(main_frame, text="数据预览", padding="10")
        data_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))

        # 创建树状视图显示数据
        self.tree = ttk.Treeview(data_frame, height=8)
        self.tree.grid(row=0, column=0, sticky="nsew")

        # 添加滚动条
        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # 变量选择区域
        var_frame = ttk.LabelFrame(main_frame, text="变量选择", padding="10")
        var_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        ttk.Label(var_frame, text="因变量 (Y):").grid(row=0, column=0, sticky="w")
        self.dependent_var = tk.StringVar()
        self.dep_combo = ttk.Combobox(var_frame, textvariable=self.dependent_var, width=30)
        self.dep_combo.grid(row=0, column=1, padx=(10, 20), sticky="w")

        ttk.Label(var_frame, text="自变量 (X):").grid(row=0, column=2, sticky="w")
        self.independent_vars = tk.StringVar()
        self.indep_combo = ttk.Combobox(var_frame, textvariable=self.independent_vars, width=30)
        self.indep_combo.grid(row=0, column=3, padx=(10, 0), sticky="w")

        # 添加变量选择按钮
        ttk.Button(var_frame, text="自动选择", command=self.auto_select_vars).grid(row=1, column=0, pady=(10, 0))
        ttk.Button(var_frame, text="手动选择", command=self.manual_select_vars).grid(row=1, column=1, pady=(10, 0))

        # 分析按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(button_frame, text="执行回归分析",
                   command=self.run_analysis, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="保存结果",
                   command=self.save_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="生成图表",
                   command=self.generate_charts).pack(side=tk.LEFT)

        # 结果展示区域
        result_frame = ttk.LabelFrame(main_frame, text="分析结果", padding="10")
        result_frame.grid(row=5, column=0, columnspan=3, sticky="nsew")

        # 创建标签页控件
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 回归系数标签页
        coef_frame = ttk.Frame(self.notebook)
        self.notebook.add(coef_frame, text="回归系数")

        self.coef_tree = ttk.Treeview(coef_frame)
        self.coef_tree.pack(fill=tk.BOTH, expand=True)

        # VIF标签页
        vif_frame = ttk.Frame(self.notebook)
        self.notebook.add(vif_frame, text="方差膨胀因子")

        self.vif_tree = ttk.Treeview(vif_frame)
        self.vif_tree.pack(fill=tk.BOTH, expand=True)

        # 模型摘要标签页
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="模型摘要")

        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=15)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # 图表标签页
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="图表")

        # 配置权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)

    def load_data(self):
        """加载数据"""
        if not self.file_path.get():
            messagebox.showerror("错误", "请先选择Excel文件")
            return

        try:
            self.df = pd.read_excel(self.file_path.get())
            self.status_var.set(f"成功加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")

            # 显示数据预览
            self.display_data_preview()

            # 更新变量选择下拉框
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.dep_combo['values'] = numeric_cols
            self.indep_combo['values'] = numeric_cols

            # 自动选择变量
            self.auto_select_vars()

        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")

    def display_data_preview(self):
        """显示数据预览"""
        # 清空现有数据
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 设置列
        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"

        # 设置列标题
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # 添加数据（只显示前50行）
        for i, row in self.df.head(50).iterrows():
            self.tree.insert("", "end", values=list(row))

    def auto_select_vars(self):
        """自动选择变量"""
        if self.df is None:
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            messagebox.showwarning("警告", "未找到数值列")
            return

        # 尝试自动识别因变量
        dep_candidates = ['销售额', '销售', '收入', '营收', 'sales', 'revenue', 'target']
        dependent_var = None

        for candidate in dep_candidates:
            for col in numeric_cols:
                if candidate.lower() in col.lower():
                    dependent_var = col
                    break
            if dependent_var:
                break

        # 如果没有找到合适的因变量，使用第一个数值列
        if not dependent_var and numeric_cols:
            dependent_var = numeric_cols[0]

        self.dependent_var.set(dependent_var)

        # 自变量：除了因变量外的所有数值列
        independent_vars = [col for col in numeric_cols if col != dependent_var]
        self.independent_vars.set(", ".join(independent_vars))

        self.status_var.set("已自动选择变量")

    def manual_select_vars(self):
        """手动选择变量对话框"""
        if self.df is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # 创建选择窗口
        select_window = tk.Toplevel(self.root)
        select_window.title("选择变量")
        select_window.geometry("400x500")
        select_window.transient(self.root)
        select_window.grab_set()

        ttk.Label(select_window, text="因变量 (Y):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))

        dep_var = tk.StringVar()
        if self.dependent_var.get():
            dep_var.set(self.dependent_var.get())

        dep_frame = ttk.Frame(select_window)
        dep_frame.pack(fill="x", padx=10)

        for col in numeric_cols:
            ttk.Radiobutton(dep_frame, text=col, variable=dep_var, value=col).pack(anchor="w")

        ttk.Label(select_window, text="自变量 (X):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))

        indep_frame = ttk.Frame(select_window)
        indep_frame.pack(fill="both", expand=True, padx=10)

        # 创建滚动框架用于自变量选择
        canvas = tk.Canvas(indep_frame, height=200)
        scrollbar = ttk.Scrollbar(indep_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 自变量选择状态
        indep_vars = {}
        current_indep = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]

        for col in numeric_cols:
            if col == dep_var.get():
                continue
            var = tk.BooleanVar()
            var.set(col in current_indep)
            indep_vars[col] = var
            ttk.Checkbutton(scrollable_frame, text=col, variable=var).pack(anchor="w")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def apply_selection():
            self.dependent_var.set(dep_var.get())

            selected_indep = [col for col, var in indep_vars.items() if var.get()]
            self.independent_vars.set(", ".join(selected_indep))

            select_window.destroy()
            self.status_var.set("已手动选择变量")

        ttk.Button(select_window, text="应用", command=apply_selection).pack(pady=10)

    def run_analysis(self):
        """执行回归分析"""
        if self.df is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        dependent_var = self.dependent_var.get()
        independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]

        if not dependent_var:
            messagebox.showerror("错误", "请选择因变量")
            return

        if not independent_vars:
            messagebox.showerror("错误", "请选择至少一个自变量")
            return

        try:
            self.status_var.set("正在执行回归分析...")
            self.root.update()

            # 准备数据
            X = self.df[independent_vars]
            y = self.df[dependent_var]

            # 处理缺失值
            X = X.dropna()
            y = y.dropna()

            # 添加常数项
            X_with_const = sm.add_constant(X)

            # 执行OLS回归
            model = sm.OLS(y, X_with_const)
            self.results = model.fit()

            # 显示结果
            self.display_results()

            self.status_var.set("回归分析完成")

        except Exception as e:
            messagebox.showerror("错误", f"回归分析失败: {str(e)}")
            self.status_var.set("分析失败")

    def display_results(self):
        """显示分析结果"""
        if self.results is None:
            return

        # 显示回归系数
        self.display_coefficients()

        # 显示VIF
        self.display_vif()

        # 显示模型摘要
        self.display_summary()

    def display_coefficients(self):
        """显示回归系数"""
        # 清空现有数据
        for item in self.coef_tree.get_children():
            self.coef_tree.delete(item)

        # 设置列
        self.coef_tree["columns"] = ["变量", "系数", "标准误", "T值", "P值"]
        self.coef_tree["show"] = "headings"

        # 设置列标题
        for col in self.coef_tree["columns"]:
            self.coef_tree.heading(col, text=col)
            self.coef_tree.column(col, width=100)

        # 添加数据
        independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]
        all_vars = ['常数项'] + independent_vars

        for i, var in enumerate(all_vars):
            values = [
                var,
                f"{self.results.params[i]:.4f}",
                f"{self.results.bse[i]:.4f}",
                f"{self.results.tvalues[i]:.4f}",
                f"{self.results.pvalues[i]:.4f}"
            ]
            self.coef_tree.insert("", "end", values=values)

    def display_vif(self):
        """显示方差膨胀因子"""
        # 清空现有数据
        for item in self.vif_tree.get_children():
            self.vif_tree.delete(item)

        # 设置列
        self.vif_tree["columns"] = ["变量", "VIF"]
        self.vif_tree["show"] = "headings"

        # 设置列标题
        for col in self.vif_tree["columns"]:
            self.vif_tree.heading(col, text=col)
            self.vif_tree.column(col, width=200)

        # 计算VIF
        independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]
        X = self.df[independent_vars]
        X_with_const = sm.add_constant(X)

        vif_data = []
        for i in range(X_with_const.shape[1]):
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_data.append((X_with_const.columns[i], vif))

        # 添加数据
        for var, vif in vif_data:
            self.vif_tree.insert("", "end", values=[var, f"{vif:.2f}"])

    def display_summary(self):
        """显示模型摘要"""
        self.summary_text.delete(1.0, tk.END)

        summary_text = f"""回归分析结果摘要

模型统计:
R²: {self.results.rsquared:.4f}
调整R²: {self.results.rsquared_adj:.4f}
F统计量: {self.results.fvalue:.4f}
F统计量P值: {self.results.f_pvalue:.4f}

样本数量: {self.results.nobs}
变量数量: {len(self.results.params)}

显著变量 (P值 < 0.05):
"""

        # 找出显著变量
        independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]
        all_vars = ['常数项'] + independent_vars

        significant_found = False
        for i, var in enumerate(all_vars):
            if self.results.pvalues[i] < 0.05:
                significant_found = True
                effect = "增加" if self.results.params[i] > 0 else "减少"
                summary_text += f"- {var}: 每单位变化使因变量{effect} {abs(self.results.params[i]):.2f} (P值: {self.results.pvalues[i]:.4f})\n"

        if not significant_found:
            summary_text += "无显著变量\n"

        # 共线性诊断
        summary_text += f"\n共线性诊断:\n"

        # 计算平均VIF（排除常数项）
        independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]
        X = self.df[independent_vars]
        X_with_const = sm.add_constant(X)

        vif_values = []
        for i in range(1, X_with_const.shape[1]):  # 跳过常数项
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_values.append(vif)

        if vif_values:
            avg_vif = np.mean(vif_values)
            max_vif = max(vif_values)

            summary_text += f"平均VIF: {avg_vif:.2f}\n"
            summary_text += f"最大VIF: {max_vif:.2f}\n"

            if max_vif > 10:
                summary_text += "⚠️ 存在严重多重共线性问题\n"
            elif max_vif > 5:
                summary_text += "⚠️ 存在中等多重共线性问题\n"
            else:
                summary_text += "✅ 多重共线性问题轻微\n"

        self.summary_text.insert(1.0, summary_text)

    def generate_charts(self):
        """生成图表"""
        if self.df is None or self.results is None:
            messagebox.showerror("错误", "请先执行回归分析")
            return

        try:
            self.status_var.set("正在生成图表...")
            self.root.update()

            # 清空图表框架
            for widget in self.chart_frame.winfo_children():
                widget.destroy()

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle("回归分析图表", fontsize=16)

            dependent_var = self.dependent_var.get()
            independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]

            # 1. 实际值 vs 预测值
            y_pred = self.results.fittedvalues
            axes[0, 0].scatter(self.results.fittedvalues, self.df[dependent_var], alpha=0.6)
            axes[0, 0].plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('预测值')
            axes[0, 0].set_ylabel('实际值')
            axes[0, 0].set_title('实际值 vs 预测值')

            # 2. 残差图
            residuals = self.results.resid
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('预测值')
            axes[0, 1].set_ylabel('残差')
            axes[0, 1].set_title('残差图')

            # 3. 系数图
            coef_vars = independent_vars
            coef_values = self.results.params[1:].values  # 排除常数项

            # 只显示前10个系数
            if len(coef_vars) > 10:
                coef_vars = coef_vars[:10]
                coef_values = coef_values[:10]

            y_pos = np.arange(len(coef_vars))
            axes[1, 0].barh(y_pos, coef_values, align='center')
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(coef_vars)
            axes[1, 0].set_xlabel('系数值')
            axes[1, 0].set_title('回归系数')

            # 4. 变量相关性热力图
            if len(independent_vars) > 1:
                corr_matrix = self.df[independent_vars].corr()
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                axes[1, 1].set_xticks(range(len(independent_vars)))
                axes[1, 1].set_xticklabels(independent_vars, rotation=45, ha='right')
                axes[1, 1].set_yticks(range(len(independent_vars)))
                axes[1, 1].set_yticklabels(independent_vars)
                plt.colorbar(im, ax=axes[1, 1])
                axes[1, 1].set_title('变量相关性')
            else:
                axes[1, 1].text(0.5, 0.5, '需要至少两个变量\n才能显示相关性矩阵',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('变量相关性')

            plt.tight_layout()

            # 在Tkinter中显示图表
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 添加工具栏
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("图表生成完成")

        except Exception as e:
            messagebox.showerror("错误", f"生成图表失败: {str(e)}")
            self.status_var.set("图表生成失败")

    def save_results(self):
        """保存结果"""
        if self.results is None:
            messagebox.showerror("错误", "没有可保存的结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            self.status_var.set("正在保存结果...")
            self.root.update()

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 保存回归系数
                independent_vars = [x.strip() for x in self.independent_vars.get().split(",") if x.strip()]
                all_vars = ['常数项'] + independent_vars

                coef_data = {
                    '变量': all_vars,
                    '系数': self.results.params,
                    '标准误': self.results.bse,
                    'T值': self.results.tvalues,
                    'P值': self.results.pvalues
                }
                coef_df = pd.DataFrame(coef_data)
                coef_df.to_excel(writer, sheet_name='回归系数', index=False)

                # 保存VIF
                X = self.df[independent_vars]
                X_with_const = sm.add_constant(X)

                vif_data = []
                for i in range(X_with_const.shape[1]):
                    vif = variance_inflation_factor(X_with_const.values, i)
                    vif_data.append((X_with_const.columns[i], vif))

                vif_df = pd.DataFrame(vif_data, columns=['变量', 'VIF'])
                vif_df.to_excel(writer, sheet_name='方差膨胀因子', index=False)

                # 保存模型摘要
                summary_data = {
                    '统计量': ['R²', '调整R²', 'F统计量', 'F统计量P值', '样本数量'],
                    '值': [
                        self.results.rsquared,
                        self.results.rsquared_adj,
                        self.results.fvalue,
                        self.results.f_pvalue,
                        self.results.nobs
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='模型摘要', index=False)

            self.status_var.set(f"结果已保存: {file_path}")
            messagebox.showinfo("成功", f"分析结果已保存到:\n{file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")
            self.status_var.set("保存失败")


def main():
    root = tk.Tk()
    app = RegressionAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()