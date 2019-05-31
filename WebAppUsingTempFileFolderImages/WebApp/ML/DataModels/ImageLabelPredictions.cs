

using Microsoft.ML.Data;
using static WebApp.ML.TensorFlowModelConfigurator;

namespace WebApp.ML.DataModels
{
    public class ImageLabelPredictions
    {
        //TODO: Change to fixed output column name for TensorFlow model
        [ColumnName(TensorFlowModelSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}
