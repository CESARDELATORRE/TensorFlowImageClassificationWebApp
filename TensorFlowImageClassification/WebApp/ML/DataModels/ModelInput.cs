using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApp.ML.DataModels
{
    public class ModelInput
    {
        [ColumnName("Label")]
        public bool IsToxic { get; set; }


        [ColumnName("Text")]
        public string SentimentText { get; set; }
    }
}
