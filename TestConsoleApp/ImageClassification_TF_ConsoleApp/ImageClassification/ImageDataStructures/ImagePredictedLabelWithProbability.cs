using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification.ImageDataStructures
{
    public class ImagePredictedLabelWithProbability
    {
        public string ImagePath;

        public string PredictedLabel;
        public float Probability { get; set; }
    }
}
