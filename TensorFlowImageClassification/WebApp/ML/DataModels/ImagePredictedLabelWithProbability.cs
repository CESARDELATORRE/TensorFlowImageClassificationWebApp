
namespace WebApp.ML.DataModels
{
    public class ImagePredictedLabelWithProbability
    {
        public string ImagePath;

        public string PredictedLabel;
        public float Probability { get; set; }

        public long PredictionExecutionTime;
    }
}
