import torch

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    # 입력 텐서를 float32로 변환
    ground_truth = ground_truth.float()
    predictions = predictions.float()

    # 텐서를 1차원으로 평탄화
    ground_truth = ground_truth.view(-1)
    predictions = predictions.view(-1)

    # 교차 영역 계산
    intersection = (predictions * ground_truth).sum()

    # 합집합 계산
    union = predictions.sum() + ground_truth.sum()

    # Dice 계수 계산
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Dice loss 계산
    loss = 1.0 - dice

    return loss

