﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification.ImageDataStructures
{
    public class ImageInputData
    {
        public string ImagePath;

        public static IEnumerable<ImageInputData> ReadFromCsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(x => x.Split('\t'))
             .Select(x => new ImageInputData { ImagePath = Path.Combine(folder, x[0]) });
        }
    }
}
