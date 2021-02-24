using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SamplesCS.Samples
{
    public class KMode
    {
        //OGP-rect5x1,OGP-rect2x1,OGP-small5x1,OGP-circle2168
        public static OpenCvSharp.ML.KNearest GetTrainedKMode(string caprev)
        {
            var traindatas = AITrainingData.GetTrainingData(caprev); ;
            var samplex = new Mat();
            var samples = new Mat();
            samplex.ConvertTo(samples, MatType.CV_32FC1);
            var responsarray = new List<int>();
            foreach (var item in traindatas)
            {
                var tcmresizeorg = Mat.ImDecode(Convert.FromBase64String(item.TrainingImg), ImreadModes.Grayscale);

                var tcmretp = new Mat();
                tcmresizeorg.ConvertTo(tcmretp, MatType.CV_32FC1);

                //var tcmresize = new Mat();
                //Cv2.Resize(tcmretp, tcmresize, new Size(76, 76), 0, 0, InterpolationFlags.Linear);

                var stcm = tcmretp.Reshape(1, 1);
                samples.PushBack(stcm);
                responsarray.Add(item.ImgVal);
            }

            int[] rparray = responsarray.ToArray();
            var responx = new Mat(rparray.Length, 1, MatType.CV_32SC1, rparray);
            var respons = new Mat();
            responx.ConvertTo(respons, MatType.CV_32FC1);

            var kmode = OpenCvSharp.ML.KNearest.Create();
            kmode.Train(samples, OpenCvSharp.ML.SampleTypes.RowSample, respons);
            return kmode;
        }

        public static OpenCvSharp.ML.SVM GetTrainedSMode(string caprev)
        {
            var traindatas = AITrainingData.GetTrainingData(caprev); ;
            var samplex = new Mat();
            var samples = new Mat();
            samplex.ConvertTo(samples, MatType.CV_32FC1);
            var respmatx = new Mat();
            var respmat = new Mat();
            respmatx.ConvertTo(respmat, MatType.CV_32SC1);

            foreach (var item in traindatas)
            {
                var tcmresizex = Mat.ImDecode(Convert.FromBase64String(item.TrainingImg), ImreadModes.Grayscale);
                var tcmresize = new Mat();
                tcmresizex.ConvertTo(tcmresize, MatType.CV_32FC1);
                var stcm = tcmresize.Reshape(1, 1);
                samples.PushBack(stcm);
                respmat.PushBack(item.ImgVal);
            }

            var smode = OpenCvSharp.ML.SVM.Create();
            smode.Type = OpenCvSharp.ML.SVM.Types.CSvc;
            smode.KernelType = OpenCvSharp.ML.SVM.KernelTypes.Linear;
            smode.TermCriteria = TermCriteria.Both(5000, 0.000001);
            smode.Train(samples, OpenCvSharp.ML.SampleTypes.RowSample, respmat);
            return smode;
        }

        public static Mat GetOneHot(int val)
        {
            var res = new int[] { 0,0,0,0,0,0,0,0,0,0};
            var idx = val - 48;
            if (idx >= 0 && idx <= 9)
            { res[idx] = 1; }
            return new Mat(1, 10, MatType.CV_32FC1, res);
        }

        public static OpenCvSharp.ML.ANN_MLP GetTrainedANNMode(string caprev)
        {
            var traindatas = AITrainingData.GetTrainingData(caprev); ;
            var samplex = new Mat();
            var samples = new Mat();
            samplex.ConvertTo(samples, MatType.CV_32FC1);
            var respmatx = new Mat();
            var respmat = new Mat();
            respmatx.ConvertTo(respmat, MatType.CV_32FC1);

            foreach (var item in traindatas)
            {
                var tcmresizex = Mat.ImDecode(Convert.FromBase64String(item.TrainingImg), ImreadModes.Grayscale);
                var tcmresize = new Mat();
                tcmresizex.ConvertTo(tcmresize, MatType.CV_32FC1);
                var tcmresize1 = new Mat();
                Cv2.Resize(tcmresize, tcmresize1, new Size(30, 30), 0, 0, InterpolationFlags.Linear);
                var stcm = tcmresize1.Reshape(0, 1);
                samples.PushBack(stcm);
                respmat.PushBack(GetOneHot(item.ImgVal).Reshape(0,1));
            }

            var smode = OpenCvSharp.ML.ANN_MLP.Create();

            var hide =(int) Math.Sqrt(9000);

            var layarray = new int[] { 900, hide,hide,hide,hide,10 }; 
            var laysize = InputArray.Create(layarray);
            smode.SetLayerSizes(laysize);
            smode.BackpropWeightScale = 0.0001;
            smode.TermCriteria = new TermCriteria(CriteriaType.MaxIter|CriteriaType.Eps, 1000, 0.0001);

            smode.Train(samples, OpenCvSharp.ML.SampleTypes.RowSample, respmat);

            return smode;
        }


    }
}
