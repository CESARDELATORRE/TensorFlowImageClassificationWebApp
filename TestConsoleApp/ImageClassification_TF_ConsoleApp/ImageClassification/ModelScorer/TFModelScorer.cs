using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML;

using ImageClassification.ImageDataStructures;
using static ImageClassification.ModelScorer.ConsoleHelpers;
using static ImageClassification.ModelScorer.ModelHelpers;
using Microsoft.ML.Transforms;

namespace ImageClassification.ModelScorer
{
    public class TFModelScorer
    {
        private readonly string _dataLocation;
        private readonly string _imagesFolder;
        private readonly string _modelLocation;
        private readonly string _labelsLocation;
        private readonly MLContext _mlContext;
        private readonly ITransformer _mlModel;

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this._dataLocation = dataLocation;
            this._imagesFolder = imagesFolder;
            this._modelLocation = modelLocation;
            this._labelsLocation = labelsLocation;

            _mlContext = new MLContext();

            // Model creation and pipeline definition for images needs to run just once, so calling it from the constructor:
            _mlModel = CreateModel(dataLocation, imagesFolder, modelLocation);
        }

        public struct ImageSettings
        {
            public const int imageHeight = 227;
            public const int imageWidth = 227;
            public const float mean = 117;         //offsetImage
            public const bool channelsLast = true; //interleavePixelColors
        }

        public struct TensorFlowModelSettings
        {
            // For checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "Placeholder";

            // output tensor name
            public const string outputTensorName = "loss";
        }

        private ITransformer CreateModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Default parameters: image size=({ImageSettings.imageWidth},{ImageSettings.imageHeight}), image mean: {ImageSettings.mean}");

            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: TensorFlowModelSettings.inputTensorName, imageFolder: imagesFolder, inputColumnName: nameof(ImageInputData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(outputColumnName: TensorFlowModelSettings.inputTensorName, imageWidth: ImageSettings.imageWidth, imageHeight: ImageSettings.imageHeight, inputColumnName: TensorFlowModelSettings.inputTensorName))
                .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: TensorFlowModelSettings.inputTensorName, interleavePixelColors: ImageSettings.channelsLast, offsetImage: ImageSettings.mean))
                .Append(_mlContext.Model.LoadTensorFlowModel(modelLocation).
                ScoreTensorFlowModel(outputColumnNames: new[] { "loss" },
                                    inputColumnNames: new[] { "Placeholder" }, addBatchDimensionInput: false));

            ITransformer mlModel = pipeline.Fit(CreateEmptyDataView());
            return mlModel;
        }
        private IDataView CreateEmptyDataView()
        {
            //Create empty DataView. We just need the schema to call fit()
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData() { ImagePath = "" });
            IEnumerable<ImageInputData> enumerableData = list;

            var dv = _mlContext.Data.LoadFromEnumerable<ImageInputData>(list);
            return dv;
        }

        public void Predict()
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageInputData, ImageNetPrediction>(_mlModel);

            var predictions = PredictDataUsingModel(_dataLocation, _imagesFolder, _labelsLocation, predictionEngine).ToArray();
        }

        protected IEnumerable<ImagePredictedLabelWithProbability> PredictDataUsingModel(string testLocation, string imagesFolder, string labelsLocation, PredictionEngine<ImageInputData, ImageNetPrediction> predictionEngine)
        {
            ConsoleWriteHeader("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");


            var labels = ModelHelpers.ReadLabels(labelsLocation);

            /////////////////////////////////////////////////////////////////////////////////////
            // IMAGE 1
            // Predict label for "green-office-chair-test.jpg"
            var image1 = new ImageInputData { ImagePath = imagesFolder + "\\" + "green-office-chair-test.jpg" };
            var image1Probabilities = predictionEngine.Predict(image1).PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var image1BestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImagePath = image1.ImagePath,
            };
            (image1BestLabelPrediction.PredictedLabel, image1BestLabelPrediction.Probability) = GetBestLabel(labels, image1Probabilities);

            image1BestLabelPrediction.ConsoleWrite();

            yield return image1BestLabelPrediction;


            /////////////////////////////////////////////////////////////////////////////////////
            // IMAGE 2
            // Predict label for "high-metal-office-chair.jpg"
            var image2 = new ImageInputData { ImagePath = imagesFolder + "\\" + "high-metal-office-chair.jpg" };
            var image2Probabilities = predictionEngine.Predict(image2).PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var image2BestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImagePath = image2.ImagePath,
            };
            (image2BestLabelPrediction.PredictedLabel, image2BestLabelPrediction.Probability) = GetBestLabel(labels, image2Probabilities);

            image2BestLabelPrediction.ConsoleWrite();

            yield return image1BestLabelPrediction;

        }
    }
}
