﻿using ImageClassification.ModelScorer;
using System;
using System.IO;


namespace ImageClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            var assetsPath = ModelHelpers.GetAssetsPath(@"..\..\..\assets");

            var tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "custom-vision-tensorflow", "model.pb");
            var labelsTxt = Path.Combine(assetsPath, "inputs", "custom-vision-tensorflow", "labels.txt");

            //var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            //var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            var customInceptionPb = Path.Combine(assetsPath, "inputs", "inception_custom", "model_tf.pb");
            var customLabelsTxt = Path.Combine(assetsPath, "inputs", "inception_custom", "labels.txt");

            try
            {
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);
                modelScorer.Predict();

            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.Message);
            }

            ConsoleHelpers.ConsolePressAnyKey();
        }
    }
}
