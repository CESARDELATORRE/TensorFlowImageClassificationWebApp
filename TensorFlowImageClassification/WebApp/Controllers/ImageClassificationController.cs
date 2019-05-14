using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using WebApp.Infrastructure;
using WebApp.ML.DataModels;

namespace WebApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageClassificationController : ControllerBase
    {
        private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;
        private readonly ILogger<ImageClassificationController> _logger;

        public ImageClassificationController(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool, ILogger<ImageClassificationController> logger) //When using DI/IoC
        {
            // Get the ML Model Engine injected, for scoring
            _predictionEnginePool = predictionEnginePool;

            //Get other injected dependencies
            _logger = logger;
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classifyimage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();


            //Convert image stream to byte[] 
            byte[] imageData = null;

            MemoryStream image = new MemoryStream();
            await imageFile.CopyToAsync(image);
            imageData = image.ToArray();
            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);

            // DELETE FILE WHEN CLOSED

            // using (FileStream fs = new FileStream(Path.GetTempFileName(), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.RandomAccess | FileOptions.DeleteOnClose))
            // { // temp file exists }


                _logger.LogInformation($"Start processing image file...");

            //Measure execution time
            var watch = System.Diagnostics.Stopwatch.StartNew();


            //Predict the image's label (The one with highest probability)
            ImagePredictedLabelWithProbability imageLabelPrediction = null;
            //imageLabelPrediction = _modelScorer.PredictLabelForImage(imageData, imageFilePath);

            ModelInput sampleData = new ModelInput() { SentimentText = "Howdy!" };

            //Predict sentiment
            ModelOutput prediction = _predictionEnginePool.Predict(sampleData);

            //Stop measuring time
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            //imageLabelPrediction.PredictionExecutionTime = elapsedMs;

            _logger.LogInformation($"Image processed in {elapsedMs} miliseconds");

                
            //return new ObjectResult(result);
            return Ok(imageLabelPrediction);

            
        }


        // GET api/ImageClassification
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "ACK Heart beat 1", "ACK Heart beat 2" };
        }


    }
}