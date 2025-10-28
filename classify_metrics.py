import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def calculate_classification_metrics(y_true, y_pred, y_score=None, average='binary'):
    """
    计算分类任务的各项指标

    参数:
    y_true -- 真实标签
    y_pred -- 预测标签
    y_score -- 预测概率得分（用于ROC和AUC计算）
    average -- 多分类时的平均方法 ('binary', 'micro', 'macro', 'weighted')

    返回:
    包含各项指标的字典
    """
    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average)
    metrics['recall'] = recall_score(y_true, y_pred, average=average)
    metrics['f1'] = f1_score(y_true, y_pred, average=average)

    # 计算混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # 如果提供了预测概率，计算ROC和AUC
    if y_score is not None:
        if average == 'binary' or len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            metrics['roc_curve'] = (fpr, tpr)
            metrics['auc'] = auc(fpr, tpr)
        else:
            # 多分类ROC曲线处理（需要更复杂的处理）
            pass

    return metrics


def plot_roc_curve(fpr, tpr, auc_score, title='ROC Curve'):
    """
    绘制ROC曲线

    参数:
    fpr -- 假正率
    tpr -- 真正率
    auc_score -- AUC值
    title -- 图表标题
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def print_classification_report(metrics, target_names=None):
    """
    打印分类报告

    参数:
    metrics -- 包含指标的字典
    target_names -- 类别名称
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")

    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


# 使用示例
if __name__ == "__main__":
    # 示例数据（二分类）
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.6, 0.85, 0.95, 0.45])

    # 计算指标
    metrics = calculate_classification_metrics(y_true, y_pred, y_score)

    # 打印报告
    print_classification_report(metrics)

    # 绘制ROC曲线（如果可用）
    if 'roc_curve' in metrics:
        fpr, tpr = metrics['roc_curve']
        plot_roc_curve(fpr, tpr, metrics['auc'])

    # # 多分类示例
    # print("\n多分类示例:")
    # y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # y_pred_multi = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2])
    #
    # metrics_multi = calculate_classification_metrics(
    #     y_true_multi, y_pred_multi, average='macro'
    # )
    # print_classification_report(metrics_multi)