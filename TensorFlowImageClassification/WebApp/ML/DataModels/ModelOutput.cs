using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApp.ML.DataModels
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool IsToxic { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
