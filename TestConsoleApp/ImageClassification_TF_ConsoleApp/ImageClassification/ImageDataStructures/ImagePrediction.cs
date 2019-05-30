using ImageClassification.ModelScorer;
using Microsoft.ML.Data;

namespace ImageClassification.ImageDataStructures
{
    public class ImagePrediction
    {
        [ColumnName(TFModelScorer.TensorFlowModelSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}
