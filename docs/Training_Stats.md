20260411T174401_772731Z
model_type=logistic_regression  train_accuracy=0.6566  val_accuracy=0.7170  train_n=865  val_n=53
stop_loss      606
take_profit    235
vertical        77
end_of_data      1


Increase the stop loss to 10%.
stop_loss      357
take_profit    299
vertical       261
end_of_data      2

"model_type":  "logistic_regression",
    "train_accuracy":  0.4069364161849711,
    "val_accuracy":  0.11538461538461539,
    "train_n":  865,
    "val_n":  52,
    "classes":  [
                    "stop_loss",
                    "take_profit",
                    "vertical"
                ],
    "features":  {
                     "log_entry_close":  true,
                     "label_geometry":  true,
                     "intraday_range_pct":  true,
                     "log1p_volume":  true
                 },
    "classification_report_val":  {
                                      "stop_loss":  {
                                                        "precision":  0.0,
                                                        "recall":  0.0,
                                                        "f1-score":  0.0,
                                                        "support":  21.0
                                                    },
                                      "take_profit":  {
                                                          "precision":  0.02564102564102564,
                                                          "recall":  0.25,
                                                          "f1-score":  0.046511627906976744,
                                                          "support":  4.0
                                                      },
                                      "vertical":  {
                                                       "precision":  0.625,
                                                       "recall":  0.18518518518518517,
                                                       "f1-score":  0.2857142857142857,
                                                       "support":  27.0
                                                   },
                                      "accuracy":  0.11538461538461539,
                                      "macro avg":  {
                                                        "precision":  0.2168803418803419,
                                                        "recall":  0.14506172839506173,
                                                        "f1-score":  0.11074197120708747,
                                                        "support":  52.0
                                                    },
                                      "weighted avg":  {
                                                           "precision":  0.32649161735700194,
                                                           "recall":  0.11538461538461539,
                                                           "f1-score":  0.15192946588295425,
                                                           "support":  52.0
                                                       }
                                  },
    "predictions_export":  "val",
    "predictions_file":  "predictions.parquet"
}


# 3
label_entry_stride = 195
stop_loss      1217
take_profit     469
vertical        151
end_of_data       2
20260411T203227_491904Z
model_type=logistic_regression  train_accuracy=0.6586  val_accuracy=0.7264  train_n=1731  val_n=106
{
    "model_type":  "logistic_regression",
    "train_accuracy":  0.658578856152513,
    "val_accuracy":  0.7264150943396226,
    "train_n":  1731,
    "val_n":  106,
    "classes":  [
                    "stop_loss",
                    "take_profit",
                    "vertical"
                ],
    "features":  {
                     "log_entry_close":  true,
                     "label_geometry":  true,
                     "intraday_range_pct":  true,
                     "log1p_volume":  true
                 },
    "classification_report_val":  {
                                      "stop_loss":  {
                                                        "precision":  0.7264150943396226,
                                                        "recall":  1.0,
                                                        "f1-score":  0.8415300546448088,
                                                        "support":  77.0
                                                    },
                                      "take_profit":  {
                                                          "precision":  0.0,
                                                          "recall":  0.0,
                                                          "f1-score":  0.0,
                                                          "support":  7.0
                                                      },
                                      "vertical":  {
                                                       "precision":  0.0,
                                                       "recall":  0.0,
                                                       "f1-score":  0.0,
                                                       "support":  22.0
                                                   },
                                      "accuracy":  0.7264150943396226,
                                      "macro avg":  {
                                                        "precision":  0.2421383647798742,
                                                        "recall":  0.3333333333333333,
                                                        "f1-score":  0.28051001821493626,
                                                        "support":  106.0
                                                    },
                                      "weighted avg":  {
                                                           "precision":  0.5276788892844428,
                                                           "recall":  0.7264150943396226,
                                                           "f1-score":  0.6113001340344366,
                                                           "support":  106.0
                                                       }
                                  },
    "predictions_export":  "val",
    "predictions_file":  "predictions.parquet"
}



#4 Class Weight Balanced
stop_loss      1217
take_profit     469
vertical        151
end_of_data       2
C:\Users\micha\Python Projects\Sparkles\artifacts\RKLB\20260411T211736_246116Z
model_type=logistic_regression  train_accuracy=0.3715  val_accuracy=0.1132  train_n=1731  val_n=106
{
    "model_type":  "logistic_regression",
    "train_accuracy":  0.37146158290005776,
    "val_accuracy":  0.11320754716981132,
    "train_n":  1731,
    "val_n":  106,
    "classes":  [
                    "stop_loss",
                    "take_profit",
                    "vertical"
                ],
    "features":  {
                     "log_entry_close":  true,
                     "label_geometry":  true,
                     "intraday_range_pct":  true,
                     "log1p_volume":  true
                 },
    "classification_report_val":  {
                                      "stop_loss":  {
                                                        "precision":  0.0,
                                                        "recall":  0.0,
                                                        "f1-score":  0.0,
                                                        "support":  77.0
                                                    },
                                      "take_profit":  {
                                                          "precision":  0.02631578947368421,
                                                          "recall":  0.2857142857142857,
                                                          "f1-score":  0.04819277108433735,
                                                          "support":  7.0
                                                      },
                                      "vertical":  {
                                                       "precision":  0.3333333333333333,
                                                       "recall":  0.45454545454545453,
                                                       "f1-score":  0.38461538461538464,
                                                       "support":  22.0
                                                   },
                                      "accuracy":  0.11320754716981132,
                                      "macro avg":  {
                                                        "precision":  0.11988304093567252,
                                                        "recall":  0.24675324675324672,
                                                        "f1-score":  0.14426938523324065,
                                                        "support":  106.0
                                                    },
                                      "weighted avg":  {
                                                           "precision":  0.07092022509102945,
                                                           "recall":  0.11320754716981132,
                                                           "f1-score":  0.0830083760295172,
                                                           "support":  106.0
                                                       }
                                  },
    "predictions_export":  "val",
    "predictions_file":  "predictions.parquet"
}