using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using SampleBase;
using OpenCvSharp.Extensions;
using System.IO;
using SamplesCS.Samples;
using OpenCvSharp.Dnn;
using OpenCvSharp.DnnSuperres;

//using Tesseract;

namespace SamplesCS
{
    /// <summary>
    /// 
    /// </summary>
    class MatOperations : ISample
    {
        public void Run()
        {
            //Run3();

            var flist = new List<string>();
            //flist.Add(@"E:\video\FAIL\f5x1\Die-3.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-16.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-22.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-27.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-30.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-37.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-54.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-16.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-21.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-56.bmp");
            //flist.Add(@"E:\video\FAIL\f5x1\Die-86.bmp");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-100.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-101.BMP");
            ////flist.Add(@"E:\video\FAIL\f5x2\DIE-102.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-103.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-104.BMP");
            ////flist.Add(@"E:\video\FAIL\f5x2\DIE-105.BMP");
            ////flist.Add(@"E:\video\FAIL\f5x2\DIE-106.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-111.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-113.BMP");
            ////flist.Add(@"E:\video\FAIL\f5x2\DIE-114.BMP");
            ////flist.Add(@"E:\video\FAIL\f5x2\DIE-117.BMP");
            //////flist.Add(@"E:\video\FAIL\f5x2\DIE-118.BMP");
            //////flist.Add(@"E:\video\FAIL\f5x2\DIE-119.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-120.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-121.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-126.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-127.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-128.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-5.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-93.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-94.BMP");
            //flist.Add(@"E:\video\FAIL\f5x2\DIE-95.BMP");

            //flist.Add(@"E:\video\FAIL\ag\1\Die-3.bmp");
            //flist.Add(@"E:\video\FAIL\ag\1\Die-67.bmp");
            //////flist.Add(@"E:\video\FAIL\ag\1\Die-71.bmp");

            //flist.Add(@"E:\video\FAIL\ag\2\Die-188.bmp");

            //////flist.Add(@"E:\video\FAIL\ag\Die-1.bmp");

            ////flist.Add(@"E:\video\FAIL\39\die-10.bmp");
            //flist.Add(@"E:\video\FAIL\39\die-35.bmp");
            //flist.Add(@"E:\video\FAIL\39\die-40.bmp");
            //flist.Add(@"E:\video\FAIL\39\die-55.bmp");
            //flist.Add(@"E:\video\FAIL\39\die-125.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-51.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-24.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-11.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-137.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-73.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-128.bmp");
            ////flist.Add(@"E:\video\FAIL\316\die-14.bmp");
            //flist.Add(@"E:\video\FAIL\316\die-15.bmp");

            //flist.Add(@"E:\video\FAIL\43\Die-16.bmp");
            //flist.Add(@"E:\video\FAIL\43\Die-21.bmp");
            //flist.Add(@"E:\video\FAIL\43\Die-49.bmp");
            //flist.Add(@"E:\video\FAIL\43\Die-64.bmp");
            //flist.Add(@"E:\video\FAIL\43\Die-112.bmp");
            //flist.Add(@"E:\video\FAIL\43\Die-159.bmp");
            //flist.Add(@"E:\video\FAIL\48\Die-111.bmp");

            //flist.Add(@"E:\video\FAIL\424\v1.png");
            //////flist.Add(@"E:\video\FAIL\424\v2.png");
            //flist.Add(@"E:\video\FAIL\424\v3.png");
            ////flist.Add(@"E:\video\FAIL\424\v4.png");
            //flist.Add(@"E:\video\FAIL\424\v5.png");

            //flist.Add(@"E:\video\FAIL\424\20200422\7b0e3ec93fbc4565b9aa7304b3e2e9cb.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\e68fa23a426642e3b3e9ab452da2b5f7.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\1df9d63c6242420389e695bc10d7b649.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\3cf44295448646449384ace378727ff9.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\9d630dffc9604d60bb6e88ab300e23b7.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\cd57678b646e4b3d80082420cc738505.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\00e80c92b8614e44a28c5d59fedaf165.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\0eaaebe9dc40480183f297f72cd6332a.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\aaf9497551d146b3872c4bb126ca6fdb.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\dee213d987b24e21b313c7d32f7f88cf.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\815b78c10388473a9152d852e038765d.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\3add985630204ea58f5720f0c7da9efd.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\f4d6a136e6fc425a9e730d646439815c.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\11125d8c92044069af16e024a237bb11.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\87c8254518c94e38a9388048aeab5aa6.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\572b153ed113490d9c0656028bd2964a.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\448dbe9064bc46b59014d5ad01a52f3f.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\3c8c30caf1b2470989640fbe7d100080.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\ce0e6c44471c43d7b9fd1db84191de0d.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\6fc06887815a42b6be6de3191d04cf58.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\1ff6757116a34523b05357e32cfdb5a5.png");

            //flist.Add(@"E:\video\FAIL\424\20200422\19b00a167c2d434cba37a98b0f0bf69d.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\8f367ae62d734982a3f74a7b701dd171.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\168cb4372e69455780d625bf73e04c59.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\1013b01c4921413d99af4509bd2599d9.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\fae84272e19046c5881c3e3a49a6e5fc.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\ef278c411bb74ab8886e2c091cf24ab1.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\706c196126fe438f994bccb2c6255ff7.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\b5d9bc60c3b44fb893d6fbac5ac2fb36.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\3944dca9557d4cd99c7d93edd8d2a1ff.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\3f79bf37f2854153997b9cc6b7fc31d2.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\205f257eda3c4478a813280036261eff.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\e61cf4f95245427db7c9f6437304078b.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\4df9e18444ce458fbca1c7fa91784305.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\c3b34fcddb384142a03f87f897f4d00b.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\080497269ddb41f18a4d3f1cb7e451a2.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\334ae47f2dd74789958f00d0afbcef54.png");
            ////flist.Add(@"E:\video\FAIL\424\20200422\12ce737095854c5f993038526978f058.png");
            //flist.Add(@"E:\video\FAIL\424\20200422\c0e14410e8da4169986083024ffad5ba.png");



            ////flist.Add(@"E:\video\5x1research\1557b53ad76c4bdcbce480ba8f229d34.png");
            //flist.Add(@"E:\video\5x1research\36910cacd8bc4ef19d50bbfd2325b9b4.png");
            //flist.Add(@"E:\video\5x1research\3698deb43d83438d8dd7e6989d696e97.png");
            //flist.Add(@"E:\video\5x1research\37d3a20594804a208e1791e5f6005ef9.png");
            //flist.Add(@"E:\video\5x1research\3d6e42c0559e4cb19bf0531cffe96c34.png");
            //flist.Add(@"E:\video\5x1research\3e7d4393581c46deb4819ddb082a27da.png");
            //flist.Add(@"E:\video\5x1research\4062d9e475e042ee8b3fdd27567188a8.png");
            //flist.Add(@"E:\video\5x1research\454484f6e2c44ff79daaa6082e2a0741.png");
            //flist.Add(@"E:\video\5x1research\52b457a6a8ef4613a77950799c48eef6.png");
            //flist.Add(@"E:\video\5x1research\6aff265df049425597881329c679d19c.png");
            ////flist.Add(@"E:\video\5x1research\73dbfd889a664811a9e3c41cb272f304.png");
            //flist.Add(@"E:\video\5x1research\783c461ccc9e45cb99fda31757551c38.png");
            //flist.Add(@"E:\video\5x1research\7fcf050e03f24fffa4ceed28b6c85057.png");
            //flist.Add(@"E:\video\5x1research\8af9b3746ada4207b1ad141a28e8701d.png");
            //flist.Add(@"E:\video\5x1research\9191cb44da024c26a8d941f5d69e5d28.png");
            //flist.Add(@"E:\video\5x1research\b34d5cd5494e44dc84265009925b3a6f.png");
            //flist.Add(@"E:\video\5x1research\c00463f40d674d34abd4ca4e4af83ae4.png");

            //flist.Add(@"\\wux-engsys01\PlanningForCast\192928-40E\RAW\Die-117.bmp");

            //{
            //    //TFGraph gf = new TFGraph();

            //flist.Add(@"E:\video\FAIL\f5x1\train-1.png");
            //var kmode = KMode.GetTrainedANNMode("OGP-rect5x1");
            //foreach (var f in flist)
            //{
            //    Run5x1(f, kmode);
            //}

            //}
            //Run5x1(@"\\wux-engsys01\PlanningForCast\62002-781-020E\Die-2.bmp", null);
            //Run5x1(@"E:\video\5x1research\c00463f40d674d34abd4ca4e4af83ae4.png", null);
            //for (var idx = 100; idx <= 120; idx++)
            //{
            //    Run5x1(@"E:\video\iivi3\DIE-"+idx+".BMP", null);
            //}


            //var kmode = KMode.GetTrainedMode("OGP-rect5x1");
            //for (var idx = 6; idx <= 6; idx++)
            //{
            //    Run5x1(@"\\wux-engsys01\PlanningForCast\292210-20E\x-" + idx + ".BMP",null);
            //}
            //{
            //    var kmode = KMode.GetTrainedKMode("OGP-rect5x1");
            //for (var idx = 1; idx <= 20; idx++)
            //{
            //    Run5x1(@"E:\video\5x1research\292407-50T\x-" + idx + ".BMP", null);
            //}
            //}

            //Mat src = Cv2.ImRead(@"E:\video\5x1research\xyz.bmp", ImreadModes.Color);
            //var tesact = new TesseractEngine(AppDomain.CurrentDomain.BaseDirectory + "\\tessdata", "eng", EngineMode.Default);
            //var bitmap = src.ToBitmap();
            //var pix = PixConverter.ToPix(bitmap);
            //var pg = tesact.Process(pix);
            //var text = pg.GetText();

            //for (var idx = 1; idx <= 1; idx++)
            //{
            //    Run5x1(@"E:\video\5x1research\line\x" + idx + ".png", null);
            //}

            //for (var idx = 7; idx <= 7; idx++)
            //{
            //    Run5x1(@"E:\video\292928-40E\x-" + idx + ".png", null);
            //}

            //for (var idx = 1; idx <= 30; idx++)
            //{
            //    Rununiq(@"E:\video\newtype\die-" + idx + ".BMP");
            //}

            //flist.Add(@"E:\video\FAIL\61950-691-010_Center.PNG");
            //foreach (var f in flist)
            //{
            //    Rununiq(f);
            //}


            //for (var idx = 1; idx <= 9; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\Vcsel1 -"
            //    + idx.ToString() + ".BMP");
            //}

            //for (var idx = 1; idx <= 9; idx++)

            //{
            //    Runsm5x1(@"E:\video\400G\Vcsel1-1-"
            //    + idx.ToString() + ".BMP");
            //}

            //for (var idx = 1; idx <= 5; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\400gnew\Vcsel1 -"
            //    + idx.ToString() + ".BMP");
            //}

            //for (var idx = 1; idx <= 5; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\400gnew\Vcsel1-1-"
            //    + idx.ToString() + ".BMP");
            //}

            //for (var idx = 42; idx <= 42; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\2set\ogp1\Vcsel1-2-"
            //    + idx.ToString() + ".BMP");
            //}

            //for (var idx = 1; idx <= 3; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\2set\dbg\Vcsel1-2-"
            //    + idx.ToString() + ".BMP");
            //}

            //Runsm5x1("E:\\video\\400G\\2set\\v2.PNG");

            //Runsm5x1("E:\\video\\400G\\2set\\dbg\\4.jpg");
            //Runsm5x1("E:\\video\\400G\\2set\\dbg\\X3FAPYG_0.bmp");

            //Runsm5x1("E:\\video\\400G\\2set\\dbg\\Vcsel3-1-13.BMP");
            //Runsm5x1("E:\\video\\400G\\2set\\dbg\\Vcsel4-1-20.BMP");

            //for (var idx = 6; idx <= 6; idx++)
            //{
            //    Runsm5x1(@"E:\video\sm5x1\die"
            //    + idx.ToString() + ".bmp");
            //}

            //for (var idx = 13; idx <= 13; idx++)
            //{
            //    Runsm5x1(@"E:\video\sm5x1\die"
            //    + idx.ToString() + ".png");
            //}

            //flist.Add(@"E:\video\newtype\die-1.bmp");
            //flist.Add(@"E:\video\newtype\die-2.bmp");
            //flist.Add(@"E:\video\newtype\die-3.bmp");
            //flist.Add(@"E:\video\newtype\die-4.bmp");

            //flist.Add(@"E:\video\newtype\die-5.bmp");
            //flist.Add(@"E:\video\newtype\die-6.bmp");
            //flist.Add(@"E:\video\newtype\die-7.bmp");
            //flist.Add(@"E:\video\newtype\die-8.bmp");
            //foreach (var f in flist)
            //{
            //    Run6(f);
            //}

            var ratelist = new List<double>();
            var radlist = new List<double>();
            //for (var idx = 1; idx <= 1; idx++)
            //{
            //    var f = @"E:\video\newtype\die-" + idx + ".bmp";
            //    Runcircle2168(f, ratelist, radlist);
            //}
            //die6 //die7 //die9

            //for (var idx = 6; idx <= 9; idx++)
            //{
            //    var f = @"E:\video\newtype\3101\die-" + idx + ".png";
            //    Runcircle2168(f, ratelist, radlist);
            //}

            //Runcircle2168(@"E:\video\newtype\die7.png", ratelist, radlist);

            //Runcircle2168(@"E:\video\newtype\DIE-2621.BMP", ratelist, radlist);

            //Run2x1(@"E:\video\newtype\cap\DIE-157.bmp");
            //Runcircle2168(@"E:\video\newtype\cap\die-10.bmp", ratelist, radlist);
            //Run2x1(@"E:\video\newtype\cap\die-1.bmp");
            //Runcircle2168(@"E:\video\newtype\prod3\DIE-115.bmp", ratelist, radlist);

            //Run2x1(@"E:\video\newtype\cap\4.png");
            //Runcircle2168(@"E:\video\newtype\cap\1.png", ratelist, radlist);
            //Runcircle2168(@"E:\video\newtype\cap\2.png", ratelist, radlist);
            //Runcircle2168(@"E:\video\newtype\cap\3.png", ratelist, radlist);
            //Runcircle2168(@"E:\video\newtype\w111.png", ratelist, radlist);
            //Runcircle2168(@"E:\video\newtype\w112.png", ratelist, radlist);
            //Runcircle2168(@"E:\video\newtype\w113.png", ratelist, radlist);


            //Run5x1(@"E:\video\3101_3103_DieID\3-X1000.bmp", null);

            //flist.Add(@"E:\video\newtype\die-1.bmp");
            //flist.Add(@"E:\video\newtype\die-2.bmp");
            //flist.Add(@"E:\video\newtype\die-3.bmp");
            //flist.Add(@"E:\video\newtype\die-4.bmp");
            //flist.Add(@"E:\video\newtype\die-5.bmp");
            //flist.Add(@"E:\video\newtype\die-6.bmp");
            //flist.Add(@"E:\video\newtype\die-7.bmp");
            //flist.Add(@"E:\video\newtype\die-8.bmp");

            //flist.Add(@"E:\video\3101_3103_DieID\3-X1000.bmp");
            //flist.Add(@"E:\video\3101_3103_DieID\1-X1000.bmp");
            //flist.Add(@"E:\video\3101\9-X1000.bmp");
            //flist.Add(@"E:\video\3101\1-X1000.bmp");
            //flist.Add(@"E:\video\3101\3-X1000.bmp");


            //flist.Add(@"E:\video\3101\die-16.bmp");
            //flist.Add(@"E:\video\3101\die-17.bmp");
            //flist.Add(@"E:\video\3101\die-19.bmp");
            //flist.Add(@"E:\video\newtype\opt\die-6.bmp");
            //foreach (var f in flist)
            //{
            //    Runcircle2168(f, ratelist, radlist,false);
            //}



            //flist.Add(@"E:\video\FAIL\ag\4\Die-60.bmp");
            //foreach (var f in flist)
            //{
            //    Run5(f);
            //}

            //var flist = new List<string>();
            //flist.Add(@"E:\video\FAIL\x\DIE-7.BMP");
            //flist.Add(@"E:\video\FAIL\x\DIE-26.BMP");
            //flist.Add(@"E:\video\FAIL\x\DIE-136.BMP");
            //flist.Add(@"E:\video\FAIL\x\DIE-151.BMP");
            //flist.Add(@"E:\video\FAIL\x\DIE-154.BMP");
            ////flist.Add(@"E:\video\FAIL\DIE-109.BMP");
            ////flist.Add(@"E:\video\FAIL\FDIE-55.PNG");
            ////flist.Add(@"E:\video\FAIL\FDIE2.PNG");
            ////flist.Add(@"E:\video\FAIL\FDIE-2.PNG");
            ////flist.Add(@"E:\video\FAIL\fdie1.png");
            ////flist.Add(@"E:\video\FAIL\fdie3.png");
            ////flist.Add(@"E:\video\FAIL\DIE-1.BMP");

            ////flist.Add(@"E:\video\FAIL\DIE-4.BMP");
            ////flist.Add(@"E:\video\FAIL\DIE-7.BMP");
            ////flist.Add(@"E:\video\FAIL\DIE-25.BMP");
            ////flist.Add(@"E:\video\FAIL\DIE-45.BMP");

            ////flist.Add(@"E:\video\FAIL\DIE-122.BMP");
            ////flist.Add(@"E:\video\FAIL\FDIE1.PNG");


            //flist.Add(@"E:\video\2X1\new\DIE-1.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-17.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-51.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-67.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-75.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-91.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-97.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-106.BMP");
            //flist.Add(@"E:\video\2X1\new\DIE-107.BMP");

            //foreach (var f in flist)
            //{
            //    Run2x1(f);
            //}

            //for (var idx = 40; idx <= 40; idx++)
            //{
            //    Run2x1(@"E:\video\2X1\Die-"+idx+".bmp");
            //}

            //for (var idx = 11; idx <= 20; idx++)
            //{
            //    Run2x1(@"E:\video\2X1\w2x1\v" + idx + ".png");
            //}

            //SubMat();
            //RowColRangeOperation();
            //RowColOperation();

            //var fs = Directory.EnumerateFiles(@"\\wux-engsys01\PlanningForCast\61940-277-040E");
            //foreach (var f in fs)
            //{
            //    var fn = Path.GetFileNameWithoutExtension(f);
            //    var ext = Path.GetFileName(f);
            //    if (ext.ToUpper().Contains(".BMP"))
            //    {
            //        var newfn = @"E:\video\FAIL\ag\61940-277-040E\" + fn + ".PNG";
            //        Mat srcimg = Cv2.ImRead(f, ImreadModes.Color);

            //        var xyenhance = new Mat();
            //        Cv2.DetailEnhance(srcimg, xyenhance);

            //        var xyptlist = GetDetectPoint(xyenhance);

            //        var xyenhgrayresize = xyenhance.SubMat(Convert.ToInt32(xyptlist[1].Min()), Convert.ToInt32(xyptlist[1].Max())
            //            , Convert.ToInt32(xyptlist[0].Min()), Convert.ToInt32(xyptlist[0].Max()));

            //        File.WriteAllBytes(newfn, xyenhgrayresize.ToBytes());
            //    }
            //}

            //var blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
            ////var xblank = new Mat();
            ////Cv2.CvtColor(blank, xblank, ColorConversionCodes.GRAY2RGB);
            //Cv2.PutText(blank, "HELLO WORLD", new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255),2,LineTypes.Link8);
            //Cv2.PutText(blank, "SkyEye", new Point(205, 52), HersheyFonts.HersheySimplex, 0.3, new Scalar(0, 0, 0), 1, LineTypes.Link8);
            //using (new Window("blank", blank))
            //{
            //    Cv2.WaitKey();
            //}


            //blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
            ////var xblank = new Mat();
            ////Cv2.CvtColor(blank, xblank, ColorConversionCodes.GRAY2RGB);
            //Cv2.PutText(blank, "HELLO WORLD", new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(255, 0, 0), 2, LineTypes.Link8);
            //Cv2.PutText(blank, "SkyEye", new Point(205, 52), HersheyFonts.HersheySimplex, 0.3, new Scalar(0, 0, 0), 1, LineTypes.Link8);
            //using (new Window("blank1", blank))
            //{
            //    Cv2.WaitKey();
            //}


            //GetCaptureImg(@"E:\video\tescat\src\");

            //removehoriz(@"E:\video\5x1research\line\src.png");

            //DetectSplitLine(@"E:\video\5x1research\line\Die-29.BMP");

            //for (var idx = 1; idx <= 20; idx++)
            //{
            //    DetectSplitLine(@"E:\video\5x1research\292407-50T\x-" + idx + ".BMP");
            //}

            //var ratelist = new List<double>();

            //for (var idx = 100; idx <= 130; idx++)
            //{
            //    RUN_IIVI(@"E:\video\iivi\DIE-" + idx + ".BMP");
            //}

            //for (var idx = 2; idx <= 6; idx++)
            //{
            //    RUN_IIVI(@"E:\video\iivi5\" + idx + ".png");
            //}

            //var kmode = KMode.GetTrainedANNMode("OGP-iivi");
            //RUN_IIVI(@"E:\video\iivi2\DIE-4.BMP", kmode);

            //RUN_PD(@"E:\video\PD\New_3076_PD_DieID\DIE2.bmp");

            //RunXProc(@"E:\video\FAIL\f5x1\train.png");

            //for (var idx = 97; idx <= 128; idx++)
            //{
            //    RunCOGA(@"E:\video\coga\COGA003\DIE-" + idx + ".BMP",ratelist);
            //}


            //RotatePicture();

            //for (var idx = 1; idx <= 1; idx++)
            //{
            //    RunA10G(@"E:\video\10G\DIE-" + idx + ".BMP");
            //}

            //for (var idx = 7; idx <= 7; idx++)
            //{
            //    RunA10G(@"E:\video\10G\color\DIE-" + idx + ".BMP");
            //}

            //for (var idx = 124; idx <= 124; idx++)
            //{
            //    RunA10G(@"E:\video\10G\dbg\DIE-" + idx + ".BMP");
            //}

            //SRMat2();

            //for (var idx = 1; idx <= 9; idx++)
            //{
            //    RUN400G(@"E:\video\400G\400gnew\v" + idx + ".png");
            //}

            //for (var idx = 1; idx <= 5; idx++)
            //{
            //    RUN400G(@"E:\video\400G\400gnew\Vcsel1 -" + idx + ".BMP");
            //}

            //var sr = new DnnSuperResImpl("edsr", 4);
            //sr.ReadModel("./EDSR_x4.pb");

            //for (var idx = 1; idx <= 10; idx++)
            //{
            //    RUN400G(@"E:\video\400G\failcut\x-" + idx + ".png", sr);
            //}

            //for (var idx = 1; idx <= 10; idx++)
            //{
            //    Runsm5x1(@"E:\video\400G\failcut\x-" + idx + ".png");
            //}

            //for (var idx = 1; idx <= 12; idx++)
            //{
            //    RUN400G(@"E:\video\400G\2set\sample\Vcsel3-1-" + idx + ".BMP");
            //}

            //SRMat3(@"E:\video\400G\failcut\x-1.png");

            for (var idx = 8; idx <= 8; idx++)
            {
                RUNIIVI_SM(@"E:\video\iivi-sm\Die-" + idx + ".BMP");
            }

            //for (var idx = 6; idx <= 6; idx++)
            //{
            //    RUNIIVI_SM(@"E:\video\sm5x1\die"
            //    + idx.ToString() + ".bmp");
            //}
        }

        public void RUNIIVI_SM(string imgpath)
        {
            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPointsm(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            {
                var outxymat = new Mat();
                Cv2.Transpose(srcrealimg, outxymat);
                Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                srcrealimg = outxymat;
            }

            using (new Window("srcimg1", srcrealimg))
            {
                Cv2.WaitKey();
            }

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 80, 80, 115);

            if (circles.Count() > 0)
            {
                var ccl = circles[0];

                //Cv2.Circle(srcrealimg, (int)ccl.Center.X, (int)ccl.Center.Y, (int)ccl.Radius, new Scalar(0, 255, 0), 3);
                //using (new Window("srcimg2", srcrealimg))
                //{
                //    Cv2.WaitKey();
                //}

                var rat = srcrealimg.Height / ccl.Radius;

                var halfheight = srcrealimg.Height / 2;
                if (ccl.Center.Y < halfheight)
                {
                    var outxymat = new Mat();
                    Cv2.Transpose(srcrealimg, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    Cv2.Transpose(outxymat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    srcrealimg = outxymat;

                    srcgray = new Mat();
                    Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);
                    circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 80, 80, 115);
                    ccl = circles[0];
                }

                //var rat = srcrealimg.Height / ccl.Radius;
                //using (new Window("srcgray", srcgray))
                //{
                //    Cv2.WaitKey();
                //}

                var xcoordx = (int)(ccl.Center.X + 70);
                var xcoordy = (int)(ccl.Center.Y + 22);
                var ycoordx = (int)(ccl.Center.X + 6);
                var ycoordy = (int)(ccl.Center.Y - 212);

                var markx = (int)(ccl.Center.X + 100);
                var marky = (int)(ccl.Center.Y - 30);

                if (ycoordy < 0) { ycoordy = 3; }

                var ximg = srcrealimg.SubMat(new Rect(xcoordx, xcoordy, 82, 40));


                var yimg = srcrealimg.SubMat(new Rect(ycoordx, ycoordy, 40, 98));
                {
                    var outxymat = new Mat();
                    Cv2.Transpose(yimg, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    yimg = outxymat;
                }

                var combinimg = new Mat();
                Cv2.HConcat(ximg, yimg, combinimg);

                var markgrey = srcgray.SubMat(new Rect(markx, marky, 40, 40));
                var markmat = new Mat();
                Cv2.AdaptiveThreshold(markgrey, markmat, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);


                ximg = GetEnhanceEdgesm(ximg);
                yimg = GetEnhanceEdgesm(yimg);


                using (new Window("ximg", ximg))
                {
                    Cv2.WaitKey();
                }
                using (new Window("yimg", yimg))
                {
                    Cv2.WaitKey();
                }


                var charlist = new List<Mat>();
                charlist.Add(combinimg);
                charlist.Add(markmat);
                charlist.AddRange(GetCharMatsSM(ximg, 1));
                charlist.Add(markmat);
                charlist.AddRange(GetCharMatsSM(yimg, 2));

                var idx = 0;
                foreach (var cm in charlist)
                {
                    if (idx == 0)
                    {
                        //using (new Window("cmxxxxxxxxxxxxx" + idx, cm))
                        //{
                        //    Cv2.WaitKey();
                        //}
                    }
                    else
                    {
                        var tcm = new Mat();
                        cm.ConvertTo(tcm, MatType.CV_32FC1);
                        var tcmresize = new Mat();
                        Cv2.Resize(tcm, tcmresize, new Size(50, 50), 0, 0, InterpolationFlags.Linear);
                        using (new Window("cmxxxxxxxxxxxxx" + idx, tcmresize))
                        {
                            Cv2.WaitKey();
                        }

                        //if (kmode != null)
                        //{
                        //    var resultmat = new Mat();
                        //    var stcm = tcmresize.Reshape(0, 1);

                        //    var matched = new Mat();
                        //    var m = kmode.Predict(stcm, matched);//,OpenCvSharp.ML.StatModel.Flags.RawOutput);
                        //                                         //kmode.FindNearest(stcm, 7, resultmat, matched);
                        //    var matchstr = matched.Dump();
                        //    var ms = matchstr.Split(new string[] { "[", "]", "," }, StringSplitOptions.RemoveEmptyEntries);
                        //    var mstr = "";
                        //    foreach (var s in ms)
                        //    {
                        //        //mstr += UT.O2S((char)UT.O2I(s));
                        //        mstr += UT.O2S(UT.O2D(s));
                        //    }

                        //    var blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
                        //    Cv2.PutText(blank, m.ToString(), new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                        //    using (new Window("blank", blank))
                        //    {
                        //        Cv2.WaitKey();
                        //    }
                        //}

                    }
                    idx++;
                }

            }
        }

        private static List<List<double>> GetDetectPointsm(Mat mat)
        {
            var xyenhance = new Mat();
            Cv2.DetailEnhance(mat, xyenhance);

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(xyenhance, null, out kazeKeyPoints, kazeDescriptors);

            var wptlist = new List<KeyPoint>();
            for (var idx = 20; idx < mat.Width;)
            {
                var yhlist = new List<double>();
                var wlist = new List<KeyPoint>();
                foreach (var pt in kazeKeyPoints)
                {
                    if (pt.Pt.X >= (idx - 20) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        yhlist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 10 && (yhlist.Max() - yhlist.Min()) > 0.3 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 20;
            }

            var xlist = new List<double>();
            var ylist = new List<double>();
            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
                ylist.Add(pt.Pt.Y);
            }
            ret.Add(xlist);
            ret.Add(ylist);

            return ret;
        }

        public Mat GetEnhanceEdgesm(Mat xymat)
        {
            var xyenhance4x = new Mat();
            Cv2.Resize(xymat, xyenhance4x, new Size(xymat.Width * 4, xymat.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(3, 3), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            //using (new Window("edged", edged))
            //{
            //    Cv2.WaitKey();
            //}

            return edged;
        }

        public List<Mat> GetCharMatsSM(Mat xymat, int id)
        {
            var charlist = new List<Mat>();
            var ylowhigh = DetectMatYHighLow(xymat);
            var ylow = ylowhigh[0];
            var yhigh = ylowhigh[1];
            var splitxlist = GetMatSplit(xymat, ylow, yhigh);

            var mat1 = xymat.SubMat(ylow, yhigh, splitxlist[0], splitxlist[1]);

            var mat2 = xymat.SubMat(ylow, yhigh, splitxlist[1], splitxlist[2]);
            var mat3 = xymat.SubMat(ylow, yhigh, splitxlist[2], splitxlist[3]);
            var ret = new List<Mat>();
            ret.Add(mat1); ret.Add(mat2); ret.Add(mat3);
            return ret;
        }

        public int GetMatSplit_(Mat img,int start, int ylow, int yhigh)
        {
            var getfont = false;
            var sum = 0;
            var end = start - 110;
            if (end < 0) { end = 2; }

            for (var idx = start; idx > end;)
            {
                var submat = img.SubMat(ylow, yhigh, idx-2, idx);

                var zcnt = submat.CountNonZero();
                if (getfont && zcnt < 2)
                {
                    return (idx - 3);
                }

                if (zcnt < 2)
                { sum = 0; }
                else
                {
                    sum++;
                    if (sum > 10) {
                        getfont = true;
                    }
                }

                idx = idx - 2;
            }
            return end;
        }

        public int GetMatEnd(Mat img, int ylow, int yhigh)
        {
            var start = img.Width;
            var end = start - 100;
            var sum = 0;
            var es = 0;
            for (var idx = start; idx > end;)
            {
                var submat = img.SubMat(ylow, yhigh, idx - 2, idx);
                var zcnt = submat.CountNonZero();
                if (zcnt < 2)
                { sum = 0; }
                else
                {
                    sum++;
                    if (sum > 10)
                    {
                        es = idx;
                        break;
                    }
                }
                idx = idx - 2;
            }

            if (es == 0)
            { start = img.Width - 50; }
            else
            { start = es; }
            end = img.Width - 10;

            for (var idx = start; idx < end;)
            {
                var submat = img.SubMat(ylow, yhigh, idx, idx + 2);
                var zcnt = submat.CountNonZero();
                if (zcnt < 2)
                { return idx + 2; }

                idx = idx + 2;
            }

            return img.Width - 5;
        }

        public List<int> GetMatSplit(Mat img,int ylow,int yhigh)
        {
            var imgend = GetMatEnd(img, ylow, yhigh);
            var splitx1 = GetMatSplit_(img, imgend - 1, ylow, yhigh);
            var splitx2 = GetMatSplit_(img, splitx1 - 1, ylow, yhigh);
            var splitx3 = GetMatSplit_(img, splitx2 - 1, ylow, yhigh);
            var ret = new List<int>();
            ret.Add(splitx3);
            ret.Add(splitx2);
            ret.Add(splitx1);
            ret.Add(imgend);
            return ret;
        }

        public List<int> DetectMatYHighLow(Mat img)
        {
            var ylow = 0;
            var yhigh = 0;

            var xlow = (int)(img.Width * 0.25);
            var xhigh = (int)(img.Width * 0.75);
            var midy = (int)(img.Height * 0.5);
            var matheigh = img.Height;

            for (var idx = midy; idx > 4;)
            {
                var submat = img.SubMat(idx, idx + 2, xlow, xhigh);
                var zcnt = submat.CountNonZero();
                if (zcnt < 3)
                { ylow = idx - 1;break; }
                idx = idx - 2;
            }

            if (ylow == 0)
            { ylow = 4; }

            for (var idx = midy; idx < matheigh - 4;)
            {
                var submat = img.SubMat(idx, idx + 2, xlow, xhigh);
                var zcnt = submat.CountNonZero();
                if (zcnt < 3)
                { yhigh = idx - 1; break; }

                idx = idx + 2;
            }

            if (yhigh == 0)
            { yhigh = matheigh - 4; }

            var ret = new List<int>();
            ret.Add(ylow);ret.Add(yhigh);
            return ret;
        }


        public void RUN400G(string imgpath, DnnSuperResImpl sr)
        {
            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());
            //using (new Window("srcimg1", srcrealimg))
            //{
            //    Cv2.WaitKey();
            //}

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 50, 25, 50);

            if (circles.Count() == 3)
            {
                var cylist = new List<int>();

                foreach (var ccl in circles)
                {
                    //Cv2.Circle(srcrealimg, (int)ccl.Center.X, (int)ccl.Center.Y, (int)ccl.Radius, new Scalar(0, 255, 0), 3);
                    cylist.Add((int)ccl.Center.Y);
                }

                var c0 = circles[0];
                var c1 = circles[0];
                var c2 = circles[0];
                cylist.Sort();
                foreach (var ccl in circles)
                {
                    if ((int)(ccl.Center.Y) == cylist[0])
                    { c0 = ccl; }
                    if ((int)(ccl.Center.Y) == cylist[1])
                    { c1 = ccl; }
                    if ((int)(ccl.Center.Y) == cylist[2])
                    { c2 = ccl; }
                }

                var blurred = new Mat();
                Cv2.GaussianBlur(srcgray, blurred, new Size(5, 5), 0);
                var edged = new Mat();
                Cv2.Canny(blurred, edged, 50, 200, 3, true);

                //using (new Window("edged", edged))
                //{
                //    Cv2.WaitKey();
                //}

                var lines = Cv2.HoughLinesP(edged, 1, Math.PI / 180.0, 50, 80, 5);
                var filterlines = new List<LineSegmentPoint>();
                foreach (var line in lines)
                {
                    if (Math.Abs(line.P1.Y - line.P2.Y) > 90 //(c2.Center.Y - c0.Center.Y - 45)
                        && (line.P1.Y > c0.Center.Y && line.P1.Y < c2.Center.Y)
                        && (line.P2.Y > c0.Center.Y && line.P2.Y < c2.Center.Y))
                    {
                        if (c0.Center.X < c1.Center.X)
                        {
                            if (line.P1.X > c0.Center.X)
                            {
                                filterlines.Add(line);
                            }
                        }
                        else
                        {
                            if (line.P1.X < c0.Center.X)
                            {
                                filterlines.Add(line);
                            }
                        }
                    }
                }

                //foreach (var line in filterlines)
                //{
                //    Cv2.Line(srcrealimg, line.P1, line.P2, new Scalar(0, 255, 0), 3);
                //}

                //using (new Window("srcimg2", srcrealimg))
                //{
                //    Cv2.WaitKey();
                //}

                if (filterlines.Count > 0)
                {
                    var coordmat = new Mat();

                    var xlist = new List<int>();
                    foreach (var line in filterlines)
                    { xlist.Add(line.P1.X); }
                    var colstart = (int) xlist.Average();

                    if (c0.Center.X < c1.Center.X)
                    {
                        coordmat = srcrealimg.SubMat((int)c0.Center.Y+10, (int)c2.Center.Y-10, colstart-32, colstart);
                        var outxymat = new Mat();
                        Cv2.Transpose(coordmat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        Cv2.Transpose(outxymat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        Cv2.Transpose(outxymat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        coordmat = outxymat;
                    }
                    else
                    {
                        coordmat = srcrealimg.SubMat((int)c0.Center.Y+10, (int)c2.Center.Y-10, colstart, colstart+32);
                        var outxymat = new Mat();
                        Cv2.Transpose(coordmat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        coordmat = outxymat;
                    }

                    using (new Window("coordmat", coordmat))
                    {
                        Cv2.WaitKey();
                    }

                    Cv2.DetailEnhance(coordmat, coordmat);

                    //var coordmat4x = new Mat();
                    //sr.Upsample(coordmat, coordmat4x);//Show400G(coordmat4x);

                    var coordmat4y = new Mat();
                    Cv2.Resize(coordmat, coordmat4y, new Size(coordmat.Cols * 4, coordmat.Rows * 4), 0, 0, InterpolationFlags.Linear);
                    Show400G(coordmat4y);

                }//end if


            }

        }

        public void Show400G(Mat coordmat)
        {
            var coorenhance = new Mat();
            Cv2.DetailEnhance(coordmat, coorenhance);

            var blurred = new Mat();
            var coorgray = new Mat();
            var denoisemat2 = new Mat();
            Cv2.FastNlMeansDenoisingColored(coorenhance, denoisemat2, 10, 10, 7, 21);
            //Cv2.CvtColor(denoisemat2, coorgray, ColorConversionCodes.BGR2GRAY);
            //Cv2.GaussianBlur(coorgray, blurred, new Size(13, 13), 2.6);
            //var edged = new Mat();
            //Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            using (new Window("denoisemat2", denoisemat2))
            {
                Cv2.WaitKey();
            }

            var lowspec = new Scalar(0, 0, 0);
            var highspec = new Scalar(116, 106, 72);

            var coordhsv = new Mat();
            Cv2.CvtColor(denoisemat2, coordhsv, ColorConversionCodes.BGR2RGB);
            //using (new Window("coordhsv", coordhsv))
            //{
            //    Cv2.WaitKey();
            //}

            var mask = coordhsv.InRange(lowspec, highspec);
            using (new Window("mask", mask))
            {
                Cv2.WaitKey();
            }
        }


        public void SRMat3(string imgpath)
        {
            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            using (new Window("srcrealimg", srcrealimg))
            {
                Cv2.WaitKey();
            }

            var src2x = new Mat();
            var sr = new DnnSuperResImpl("edsr", 2);
            sr.ReadModel("./EDSR_x2.pb");
            sr.Upsample(srcrealimg, src2x);

            using (new Window("src2x", src2x))
            {
                Cv2.WaitKey();
            }
        }

        public Mat SRMat(Mat src)
        {
            //var src = Cv2.ImRead(@"E:\video\400G\2set\dbg\input.png");
            var net = OpenCvSharp.Dnn.Net.ReadNetFromONNX(@"E:\video\400G\2set\dbg\super-resolution-10.onnx");

            var h = src.Rows;
            var w = src.Cols;

            var srccp = new Mat();
            src.CopyTo(srccp);
            Cv2.Resize(srccp, srccp, new Size(224, 224), 0, 0, InterpolationFlags.Cubic);

            var imgycbcr = new Mat();
            Cv2.CvtColor(srccp, imgycbcr, ColorConversionCodes.BGR2YCrCb);

            //using (new Window("imgycbcr", imgycbcr))
            //{
            //    Cv2.WaitKey();
            //}

            var imgycbs = imgycbcr.Split();
            var y = imgycbs[0];

            //using (new Window("y", y))
            //{
            //    Cv2.WaitKey();
            //}

            var yshape = y.Reshape(1, new int[] {1,1, 224, 224 });
            var yshapef = new Mat();
            yshape.ConvertTo(yshapef, MatType.CV_32F, 1.0 / 255);

            net.SetInput(yshapef);
            var ret = net.Forward();

            //ret.Reshape(1,1);
            var retshape = ret.Reshape(0,new int[] { 672, 672 });
            retshape = retshape * 255.0;
            retshape.Normalize(1.0, 255.0);

            var cy = new Mat();
            retshape.ConvertTo(cy, MatType.CV_8UC1);

            //using (new Window("cy", cy))
            //{
            //    Cv2.WaitKey();
            //}

            var retstr = cy.Dump();

            var cb = new Mat();
            Cv2.Resize(imgycbs[1], cb, new Size(672, 672), 0, 0, InterpolationFlags.Cubic);
            var cr = new Mat();
            Cv2.Resize(imgycbs[2], cr, new Size(672, 672), 0, 0, InterpolationFlags.Cubic);

            var matlist = new List<Mat>();
            matlist.Add(cy);matlist.Add(cb);matlist.Add(cr);
            var finalmat = new Mat();
            Cv2.Merge(matlist.ToArray(), finalmat);

            //using (new Window("finalmat", finalmat))
            //{
            //    Cv2.WaitKey();
            //}

            var fmat = new Mat();
            Cv2.CvtColor(finalmat, fmat, ColorConversionCodes.YCrCb2BGR);

            fmat = fmat.Resize(new Size(w*4, h*4));
            using (new Window("fmat", fmat))
            {
                Cv2.WaitKey();
            }

            return fmat;
        }

        public Mat SRMat1()
        {
            var src = Cv2.ImRead(@"E:\video\400G\2set\dbg\input.png");
            var scale = 4;
            var net = OpenCvSharp.Dnn.Net.ReadNetFromTensorflow(@"E:\video\400G\2set\dbg\EDSR_x4.pb");

            var w = src.Cols - (src.Cols % scale);
            var h = src.Rows - (src.Rows % scale);
            var cropped = src.SubMat(new Rect(0, 0, w, h));
            var downscal = new Mat();
            Cv2.Resize(cropped, downscal, new Size(), 1.0 / scale, 1.0 / scale);

            var mean = new Scalar(103.1545782, 111.561547, 114.35629928);
            var fmat = new Mat();
            downscal.ConvertTo(fmat, MatType.CV_32F, 1.0);

            var blob = CvDnn.BlobFromImage(fmat, 1.0, new Size(), mean);


  
            net.SetInput(blob);
            var ret = net.Forward();

            return new Mat();
        }


        public Mat SRMat2()
        {
            var src = Cv2.ImRead(@"E:\video\400G\2set\dbg\210088.png");

            var scale = 4;
            var net = OpenCvSharp.Dnn.Net.ReadNetFromTorch(@"E:\video\400G\2set\dbg\edsr.pb");

            var w = src.Cols - (src.Cols % scale);
            var h = src.Rows - (src.Rows % scale);
            var cropped = src.SubMat(new Rect(0, 0, w, h));
            var downscal = new Mat();
            Cv2.Resize(cropped, downscal, new Size(), 1.0 / scale, 1.0 / scale);

            var mean = new Scalar(103.1545782, 111.561547, 114.35629928);
            var fmat = new Mat();
            downscal.ConvertTo(fmat, MatType.CV_32F, 1.0);

            var blob = CvDnn.BlobFromImage(fmat, 1.0, new Size(), mean);



            net.SetInput(blob);
            var ret = net.Forward();

            return new Mat();
        }

        public void RunA10G(string imgpath)
        {
            Mat srccolor = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srccolor);
            var wd = (int)detectsize[0].Max() - (int)detectsize[0].Min();
            var ht = (int)detectsize[1].Max() - (int)detectsize[1].Min();
            var wdrate = (double)wd / (double)srccolor.Width;
            var htrate = (double)ht / (double)srccolor.Height;
            var srcrealimg = srccolor;
            if (wdrate >= 0.6 && htrate >= 0.6)
            { srcrealimg = srccolor.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max()); }
            

            //using (new Window("srcrealimg", srcrealimg))
            //{
            //    Cv2.WaitKey();
            //}

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);
            var srcblurred = new Mat();
            Cv2.GaussianBlur(srcgray, srcblurred, new Size(5, 5), 0);
            var srcedged = new Mat();
            Cv2.AdaptiveThreshold(srcblurred, srcedged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            using (new Window("srcedged", srcedged))
            {
                Cv2.WaitKey();
            }

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 65, 30, 80);

            foreach (var c in circles)
            {
                Cv2.Circle(srcrealimg, (int)c.Center.X, (int)c.Center.Y, (int)c.Radius, new Scalar(0, 255, 0), 3);
                using (new Window("srcimg2", srcrealimg))
                {
                    Cv2.WaitKey();
                }
            }

            if (circles.Count() > 1)
            {
                var largecircle = circles[0];
                var smallcircle = circles[0];
                foreach (var c in circles)
                {
                    if (c.Radius > largecircle.Radius)
                    { largecircle = c; }
                    if (c.Radius < smallcircle.Radius)
                    { smallcircle = c; }
                }
                var rate = largecircle.Radius / smallcircle.Radius;
                if (rate > 1.2 && rate < 2.1 && CheckAngle(largecircle.Center,smallcircle.Center,135,315))
                {
                    var coordmat = new Mat();

                    var xl = largecircle.Center.X - largecircle.Radius - 33;
                    var xh = largecircle.Center.X + largecircle.Radius + 33;
                    if (largecircle.Center.Y < smallcircle.Center.Y)
                    {//pos
                        var yh = largecircle.Center.Y - largecircle.Radius - 19;
                        var yl = yh - 40;
                        if (yl < 0) { yl = 1; }
                        coordmat = srcrealimg.SubMat((int)yl, (int)yh, (int)xl, (int)xh);
                    }
                    else
                    {//neg
                        var yl = largecircle.Center.Y + largecircle.Radius + 19;
                        var yh = yl + 40;
                        if (yh > srcrealimg.Height)
                        { yh = srcrealimg.Height - 1; }

                        coordmat = srcrealimg.SubMat((int)yl, (int)yh, (int)xl, (int)xh);
                        var outxymat = new Mat();
                        Cv2.Transpose(coordmat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        Cv2.Transpose(outxymat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        coordmat = outxymat;
                    }

                    using (new Window("coormat", coordmat))
                    {
                        Cv2.WaitKey();
                    }

                    Get10GMats(coordmat);
                }
            }

        }

        private List<Mat> Get10GMats(Mat coordmat)
        {
            var cmatlist = new List<Mat>();

            var xyenhance4x = new Mat();
            Cv2.Resize(coordmat, xyenhance4x, new Size(coordmat.Width * 4, coordmat.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            using (new Window("xyenhance4x", xyenhance4x))
            {
                Cv2.WaitKey();
            }

            //var lowspec = new Scalar(152, 113, 72);
            //var highspec = new Scalar(216, 174, 162);

            var lowspec = new Scalar(210, 120, 100);
            var highspec = new Scalar(255, 240, 220);

            var coordhsv = new Mat();
            Cv2.CvtColor(xyenhance4x, coordhsv, ColorConversionCodes.BGR2RGB);
            var mask = coordhsv.InRange(lowspec,highspec);

            var maskcnt = mask.CountNonZero();
            if (maskcnt < 15000)
            {
                lowspec = new Scalar(152, 113, 72);
                highspec = new Scalar(216, 174, 162);
                mask = coordhsv.InRange(lowspec, highspec);
            }

            using (new Window("mask", mask))
            {
                Cv2.WaitKey();
            }
            var rectlist = Get10GRect(mask);

            cmatlist.Add(coordmat);
            foreach (var rect in rectlist)
            {
                if (rect.X < 0 || rect.Y < 0
                || ((rect.X + rect.Width) > mask.Width)
                || ((rect.Y + rect.Height) > mask.Height))
                {
                    cmatlist.Clear();
                    return cmatlist;
                }

                cmatlist.Add(mask.SubMat(rect));
            }

            return cmatlist;
        }

        private List<Rect> Get10GRect(Mat edged)
        {
            var hl = GetHeighLow10G(edged);
            var hh = GetHeighHigh10G(edged);


            var dcl = hl;//(int)(hl + (hh - hl) * 0.1);
            var dch = hh;//(int)(hh - (hh - hl) * 0.1);
            var xxh = GetXXHigh10G(edged, dcl, dch);
            var yxl = GetYXLow10G(edged, dcl, dch);



            var rectlist = new List<Rect>();

            var xxlist = GetXSplitList10G(edged, xxh, hl, hh);
            var flist = (List<int>)xxlist[0];
            var slist = (List<int>)xxlist[1];
            var y = hl - 5;
            var h = hh - hl + 7;

            if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                var left = slist[2] - fntw - 10;
                if (left < 0) { left = 1; }
                rectlist.Add(new Rect(left, y, fntw + 4, h));
                rectlist.Add(new Rect(slist[2] - 6, y, slist[1] - slist[2] + 2, h));
                rectlist.Add(new Rect(slist[1] - 6, y, slist[0] - slist[1] + 2, h));
                rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 8, h));
            }
            //else if (slist.Count == 2)
            //{
            //    var fntw = (int)flist.Average();
            //    var left = slist[1] - 2 * fntw - 14;
            //    if (left < 0) { left = 1; }
            //    rectlist.Add(new Rect(left, y, fntw + 5, h));
            //    rectlist.Add(new Rect(slist[1] - fntw - 10, y, fntw + 3, h));
            //    rectlist.Add(new Rect(slist[1] - 6, y, slist[0] - slist[1], h));
            //    rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 8, h));
            //}
            else
            {
                if ((int)xxh - 210 > 0)
                { rectlist.Add(new Rect(xxh - 210, y, 54, h)); }
                else
                { rectlist.Add(new Rect(0, y, 56, h)); }
                rectlist.Add(new Rect(xxh - 156, y, 52, h));
                rectlist.Add(new Rect(xxh - 100, y, 52, h));
                rectlist.Add(new Rect(xxh - 50, y, 54, h));
            }

            var yxlist = GetYSplitList10G(edged, yxl, hl, hh);
            flist = (List<int>)yxlist[0];
            slist = (List<int>)yxlist[1];
            if (slist.Count == 4)
            {
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(slist[0] + 5, y, slist[1] - slist[0] + 4, h));
                rectlist.Add(new Rect(slist[1] + 5, y, slist[2] - slist[1] + 4, h));

                var st = slist[2] + 6;
                var wd = slist[3] - slist[2] + 8;
                if ((st + wd) > edged.Width - 1)
                {  wd = edged.Width - 1 - st; }
                rectlist.Add(new Rect(slist[2] + 6, y, wd, h));
            }
            else if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(slist[0] + 5, y, slist[1] - slist[0] + 4, h));
                rectlist.Add(new Rect(slist[1] + 5, y, slist[2] - slist[1] + 4, h));
                //var left = slist[2] + 5;
                //if (left + fntw + 5 > edged.Width)
                //{ left = edged.Width - fntw - 5; }
                //rectlist.Add(new Rect(left, y, fntw + 5, h));
            }
            //else if (slist.Count == 2)
            //{
            //    var fntw = (int)flist.Average();
            //    rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
            //    rectlist.Add(new Rect(slist[0] + 5, y, slist[1] - slist[0] + 4, h));
            //    rectlist.Add(new Rect(slist[1] + 7, y, fntw + 7, h));
            //    var left = slist[1] + fntw + 14;
            //    if (left + fntw + 8 > edged.Width)
            //    { left = edged.Width - fntw - 8; }
            //    rectlist.Add(new Rect(left, y, fntw + 8, h));
            //}
            else
            {
                rectlist.Add(new Rect(yxl - 4, y, 56, h));
                rectlist.Add(new Rect(yxl + 50, y, 55, h));
                rectlist.Add(new Rect(yxl + 102, y, 55, h));
                if ((yxl + 210) >= (edged.Cols - 1))
                { rectlist.Add(new Rect(yxl + 156, y, edged.Cols - yxl - 156, h)); }
                else
                { rectlist.Add(new Rect(yxl + 156, y, 54, h)); }
            }

            var idx = 0;
            foreach (var rect in rectlist)
            {
                var cmat = edged.SubMat(rect);
                using (new Window("cmat" + idx++, cmat))
                {
                    Cv2.WaitKey();
                }
            }

            return rectlist;
        }

        public static int GetXDirectSplit10G(Mat edged, int start, int end, int dcl, int dch, int previous)
        {
            var ret = -1;
            for (var idx = start; idx > end; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    if (ret == -1)
                    {
                        ret = idx;
                        if (previous - idx >= 48)
                        { return ret; }
                    }
                    else
                    { return ret; }
                }
                else
                { ret = -1; }
            }
            return -1;
        }

        public static int GetYDirectSplit10G(Mat edged, int start, int end, int dcl, int dch, int previous)
        {
            var ret = -1;
            for (var idx = start; idx < end; idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    if (ret == -1)
                    {
                        ret = idx;
                        if (idx - previous >= 48)
                        { return ret; }
                    }
                    else
                    { return ret; }
                }
                else
                { ret = -1; }
            }
            return -1;
        }

        public static List<object> GetXSplitList10G(Mat edged, int xxh, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spx1 = GetXDirectSplit10G(edged, xxh - 20, xxh - 20 - fntw, hl, hh, xxh);
            if (spx1 == -1) { return ret; }
            fntw = xxh - spx1 + 1;
            if (fntw >= 18 && fntw < 35)
            { spx1 = xxh - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx1);

            var spx2 = GetXDirectSplit10G(edged, spx1 - 24, spx1 - 24 - fntw, hl, hh, spx1);
            if (spx2 == -1) { return ret; }
            fntw = spx1 - spx2;
            if (fntw >= 18 && fntw < 35)
            { spx2 = spx1 - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx2);

            var spx3 = GetXDirectSplit10G(edged, spx2 - 24, spx2 - 24 - fntw, hl, hh, spx2);
            if (spx3 == -1) { return ret; }
            fntw = spx2 - spx3;
            if (fntw >= 18 && fntw < 35)
            { spx3 = spx2 - offset; fntw = offset; }
            if (spx3 < 0) { spx3 = 6; }
            flist.Add(fntw); slist.Add(spx3);

            return ret;
        }

        public static List<object> GetYSplitList10G(Mat edged, int yxl, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spy1 = GetYDirectSplit10G(edged, yxl + 24, yxl + 24 + fntw, hl, hh, yxl);
            if (spy1 == -1) { return ret; }
            fntw = spy1 - yxl + 1;
            if (fntw >= 18 && fntw < 35)
            { spy1 = yxl + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy1);

            var spy2 = GetYDirectSplit10G(edged, spy1 + 28, spy1 + 28 + fntw, hl, hh, spy1);
            if (spy2 == -1) { return ret; }
            fntw = spy2 - spy1 + 1;
            if (fntw >= 18 && fntw < 35)
            { spy2 = spy1 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy2);

            var spy3 = GetYDirectSplit10G(edged, spy2 + 28, spy2 + 28 + fntw, hl, hh, spy2);
            if (spy3 == -1) { return ret; }
            fntw = spy3 - spy2 + 1;
            if (fntw >= 18 && fntw < 25)
            { spy3 = spy2 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy3);

            var spy4 = GetYDirectSplit10G(edged, spy3 + 28, edged.Width-2 , hl, hh, spy3);
            if (spy4 == -1) { return ret; }
            fntw = spy4 - spy3 + 1;
            if (fntw < 40)
            { return ret; }
            flist.Add(fntw); slist.Add(spy4);

            return ret;
        }

        public static int GetHeighLow10G(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.15);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.84);

            var xhl = 0;
            var yhl = 0;
            var ymidx = (int)(edged.Height * 0.5);
            for (var idx = ymidx; idx > 5; idx = idx - 2)
            {
                if (xhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhl = idx;
                    }
                }

                if (yhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhl = idx;
                    }
                }
            }

            var hl = xhl;
            if (yhl > hl)
            { hl = yhl; }

            return hl;
        }

        public static int GetHeighHigh10G(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.15);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.84);

            var xhh = 0;
            var yhh = 0;
            var ymidx = (int)(edged.Height * 0.5);
            for (var idx = ymidx; idx < edged.Height - 5; idx = idx + 2)
            {
                if (xhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhh = idx;
                    }
                }

                if (yhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhh = idx;
                    }
                }
            }

            var hh = 0;
            if (xhh > ymidx && yhh > ymidx)
            {
                if (yhh < xhh)
                { hh = yhh; }
                else
                { hh = xhh; }
            }
            else if (xhh > ymidx)
            { hh = xhh; }
            else if (yhh > ymidx)
            { hh = yhh; }
            else
            { hh = edged.Height - 5; }
            return hh;
        }

        public static int GetXXHigh10G(Mat edged, int dcl, int dch)
        {
            var ret = -1;
            var tm = 0;
            var wml = (int)(edged.Width * 0.2);
            var wmh = (int)(edged.Width * 0.5);

            for (var idx = wmh; idx > wml; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    tm++;
                    if (ret == -1)
                    { ret = idx; }
                    else if (ret != -1 && tm > 8)
                    { return ret; }
                }
                else
                { ret = -1; tm = 0; }
            }

            return -1;
        }

        public static int GetYXLow10G(Mat edged, int dcl, int dch)
        {
            var ret = -1;
            var tm = 0;
            var wml = (int)(edged.Width * 0.5);
            var wmh = (int)(edged.Width * 0.8);

            for (var idx = wml; idx < wmh; idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    tm++;
                    if (ret == -1)
                    { ret = idx; }
                    else if (ret != -1 && tm > 8)
                    { return ret; }
                }
                else
                { ret = -1; tm = 0; }
            }
            return -1;
        }


        private bool CheckAngle(Point2f P1,Point2f P2 ,double ang1,double ang2)
        {
            var degree = Math.Atan2((P2.Y - P1.Y), (P2.X - P1.X));
            var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);
            var lowspec = ang1 - 8;
            var highspec = ang1 + 8;
            if (d360 > lowspec && d360 < highspec)
            { return true; }
            lowspec = ang2 - 8;
            highspec = ang2 + 8;
            if (d360 > lowspec && d360 < highspec)
            { return true; }

            return false;
        }

        public void RotatePicture()
        {
            Mat srccolor = Cv2.ImRead(@"E:\video\newtype\DIE-262.BMP", ImreadModes.Color);
            var center = new Point2f(srccolor.Width / 2, srccolor.Height / 2);
            var m = Cv2.GetRotationMatrix2D(center, -3, 1);
            var outxymat = new Mat();
            Cv2.WarpAffine(srccolor, outxymat, m, new Size(srccolor.Width, srccolor.Height));
            srccolor = outxymat;
            srccolor.SaveImage(@"E:\video\newtype\DIE-2621.BMP");
        }


        public void RunCOGA(string imgpath,List<double> ratelist)
        {
             Mat srcrealimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            using (new Window("original", srcrealimg))
            {
                Cv2.WaitKey();
            }

            Cv2.DetailEnhance(srcrealimg, srcrealimg);
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(srcrealimg, denoisemat, 10, 10, 7, 21);


            var srcgray = new Mat();
            Cv2.CvtColor(denoisemat, srcgray, ColorConversionCodes.BGR2GRAY);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows/8,100,80);

            var Cwl = (int)(srcgray.Width * 3);
            var Cwh = (int)(srcgray.Width * 0.7);

            var Ce0 = 15;
            var Ce1 = srcgray.Width - 15;

            var Chl = (int)(srcgray.Height * 0.333);
            var Chh = (int)(srcgray.Height * 0.666);

            var filtercircle = new List<CircleSegment>();

            foreach (var ccl in circles)
            {
                //var ccl = circles[0];
                if ((ccl.Center.X < Cwl || ccl.Center.X > Cwh)
                    && (ccl.Center.X > Ce0 && ccl.Center.X < Ce1)
                    && (ccl.Center.Y > Chl && ccl.Center.Y < Chh))
                {
                    //Cv2.Circle(denoisemat, (int)ccl.Center.X, (int)ccl.Center.Y, (int)ccl.Radius, new Scalar(0, 255, 0), 3);
                    filtercircle.Add(ccl);
                }
            }

            if (filtercircle.Count > 0)
            {
                var blurred = new Mat();
                Cv2.GaussianBlur(srcgray, blurred, new Size(3, 3), 0);

                //var edged = new Mat();
                //Cv2.Canny(blurred, edged, 50, 200, 3, false);

                var edged = new Mat();
                Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

                //var edged = new Mat();
                //Cv2.Threshold(blurred, edged, 50, 200, ThresholdTypes.BinaryInv);

                //using (new Window("edged", edged))
                //{
                //    Cv2.WaitKey();
                //}


                var linedetc = OpenCvSharp.XImgProc.FastLineDetector.Create(60);
                var lines = linedetc.Detect(edged);
                var cc = filtercircle[0].Center;

                var lyl = cc.Y - 75;
                var lyh = cc.Y + 75;
                var filterlines = new List<Vec4f>();

                var vlines = new List<Vec4f>();
                var hlines = new List<Vec4f>();

                foreach (var l in lines)
                {
                    if (l.Item1 > lyl && l.Item1 < lyh && l.Item3 > lyl && l.Item3 < lyh
                        && Math.Abs(l.Item0 - cc.X) < srcgray.Width / 2 && Math.Abs(l.Item2 - cc.X) < srcgray.Width / 2)
                    {
                        filterlines.Add(l);
                        if (Math.Abs(l.Item0 - l.Item2) > 30)
                        { hlines.Add(l); }
                        else
                        { vlines.Add(l); }
                    }
                }

                linedetc.DrawSegments(srcgray, filterlines, true);

                //using (new Window("srcgray", srcgray))
                //{
                //    Cv2.WaitKey();
                //}

                if (vlines.Count > 0 && hlines.Count > 0)
                {
                    //vlines.Sort(delegate (Vec4f obj1,Vec4f obj2) {
                    //   return obj1.Item0.CompareTo(obj2.Item0);
                    //});

                    //hlines.Sort(delegate (Vec4f obj1, Vec4f obj2) {
                    //    return obj1.Item1.CompareTo(obj2.Item1);
                    //});

                    var regionmidx = (int)((hlines[0].Item0 + hlines[0].Item2) / 2);
                    var regionmidy = (int)cc.Y;

                    var upper = -1;
                    for (var idx = regionmidy; idx > 0; idx--)
                    {
                        foreach (var l in hlines)
                        {
                            if (l.Item1 < regionmidy && upper == -1 && l.Item1 >= idx) {
                                upper = (int)l.Item1;
                                break;
                            }
                        }
                        if (upper != -1) { break; }
                    }//end for

                    var botm = -1;
                    for (var idx = regionmidy; idx < srcgray.Height; idx++)
                    {
                        foreach (var l in hlines)
                        {
                            if (l.Item1 > regionmidy && botm == -1 && l.Item1 <= idx)
                            {
                                botm = (int)l.Item1;
                                break;
                            }
                        }
                        if (botm != -1) { break; }
                    }//end for

                    var left = -1;
                    for (var idx = regionmidx; idx > 0; idx--)
                    {
                        foreach (var l in vlines)
                        {
                            if (l.Item0 < regionmidx && left == -1 && l.Item0 >= idx)
                            {
                                left = (int)l.Item0;
                                break;
                            }
                        }
                        if (left != -1) { break; }
                    }//end for

                    var right = -1;
                    for (var idx = regionmidx; idx < srcgray.Width; idx++)
                    {
                        foreach (var l in vlines)
                        {
                            if (l.Item0 > regionmidx && right == -1 && l.Item0 <= idx)
                            {
                                right = (int)l.Item0;
                                break;
                            }
                        }
                        if (right != -1) { break; }
                    }//end for

                    if (upper != -1 && botm != -1)
                    {

                        //AVG 105,  >90  <120
                        ratelist.Add(botm - upper);

                        if (left != -1 && right == -1)
                        {
                            right = left + 150;
                        }
                        else if (left == -1 && right != -1)
                        {
                            left = right - 150;
                        }
                        else if (left == -1 && right == -1)
                        {
                            left = regionmidx - 75;
                            right = regionmidx + 75;
                        }

                        var coordmat = srcrealimg.SubMat(upper, botm, left, right);
                        if (cc.X < regionmidx)
                        {
                            var outxymat = new Mat();
                            Cv2.Transpose(coordmat, outxymat);
                            Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                            Cv2.Transpose(outxymat, outxymat);
                            Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                            coordmat = outxymat;
                        }

                        using (new Window("coordmat", coordmat))
                        {
                            Cv2.WaitKey();
                        }

                        Cv2.DetailEnhance(coordmat, coordmat);

                        var xyenhance4x = new Mat();
                        Cv2.Resize(coordmat, xyenhance4x, new Size(coordmat.Width * 1.6, coordmat.Height * 1.6));

                        
                        Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

                        var xyenhgray = new Mat();
                        var denoisemat1 = new Mat();
                        Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat1, 10, 10, 7, 21);
                        Cv2.CvtColor(denoisemat1, xyenhgray, ColorConversionCodes.BGR2GRAY);


                        var blurred1 = new Mat();
                        Cv2.GaussianBlur(xyenhgray, blurred1, new Size(13, 13), 3.2);

                        var coordedged = new Mat();
                        Cv2.AdaptiveThreshold(blurred1, coordedged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);


                        //using (new Window("edged1", coordedged))
                        //{
                        //    Cv2.WaitKey();
                        //}

                        var xl = (int)(coordedged.Width * 0.35);
                        var xh = (int)(coordedged.Width * 0.65);
                        var xmidy = (int)(coordedged.Height * 0.25);

                        var xupper = GetCoordUpper(coordedged,xl, xh, xmidy, 12);
                        var xmaxy = (int)(coordedged.Height * 0.5 + 20);
                        var xbotm = GetCoordBotm(coordedged, xl, xh, xmidy, xmaxy);

                        var ymidy = (int)(coordedged.Height * 0.75);
                        var yupper = GetCoordUpper(coordedged, xl, xh, ymidy, xbotm);
                        var ybotm = GetCoordBotm(coordedged, xl, xh, ymidy, coordedged.Height-12);


                        var xposlist = new List<int>();

                        try
                        {
                            var xc = 26;
                            var step = (int)(coordedged.Width * 0.22);
                            var coordstart = GetFirstPos(coordedged, 14, step , xupper, xbotm, yupper, ybotm);
                            if (coordstart != -1)
                            {
                                xposlist.Add(coordstart);
                                var xpos1 = GetXPos(coordedged, coordstart + xc, coordstart + step + xc, xupper, xbotm, yupper, ybotm);
                                if (xpos1 != -1)
                                {
                                    xposlist.Add(xpos1);
                                    var xpos2 = GetXPos(coordedged, xpos1 + xc, xpos1 + step + xc, xupper, xbotm, yupper, ybotm);
                                    if (xpos2 != -1)
                                    {
                                        xposlist.Add(xpos2);
                                        var xpos3 = GetXPos(coordedged, xpos2 + xc, xpos2 + step + xc, xupper, xbotm, yupper, ybotm);
                                        if (xpos3 != -1)
                                        { xposlist.Add(xpos3); }
                                    }//end xpos2
                                }//end xpos1
                            }//end coordstart
                        }
                        catch (Exception ex) { xposlist.Clear(); }


                        var cmlist = new List<Mat>();

                        if (xposlist.Count > 0)
                        {
                            if (xposlist.Count == 4)
                            {
                                var fntlist = new List<int>();
                                fntlist.Add(xposlist[1] - xposlist[0]);
                                fntlist.Add(xposlist[2] - xposlist[1]);
                                fntlist.Add(xposlist[3] - xposlist[2]);
                                var fntwd = (int)fntlist.Average() - 2;
                                var xend = xposlist[3] + fntwd;
                                if (coordedged.Width < xend)
                                { xend = coordedged.Width - 3; }

                                var x1 = coordedged.SubMat(xupper, xbotm, xposlist[0], xposlist[1]);
                                var x2 = coordedged.SubMat(xupper, xbotm, xposlist[1], xposlist[2]);
                                var x3 = coordedged.SubMat(xupper, xbotm, xposlist[2], xposlist[3]);
                                var x4 = coordedged.SubMat(xupper, xbotm, xposlist[3], xend);

                                var y1 = coordedged.SubMat(yupper, ybotm, xposlist[0], xposlist[1]);
                                var y2 = coordedged.SubMat(yupper, ybotm, xposlist[1], xposlist[2]);
                                var y3 = coordedged.SubMat(yupper, ybotm, xposlist[2], xposlist[3]);
                                var y4 = coordedged.SubMat(yupper, ybotm, xposlist[3], xend);

                                cmlist.Add(x1); cmlist.Add(x2); cmlist.Add(x3); cmlist.Add(x4);
                                cmlist.Add(y1); cmlist.Add(y2); cmlist.Add(y3); cmlist.Add(y4);
                            }
                            else if (xposlist.Count == 3)
                            {
                                var fntlist = new List<int>();
                                fntlist.Add(xposlist[1] - xposlist[0]);
                                fntlist.Add(xposlist[2] - xposlist[1]);
                                var fntwd = (int)fntlist.Average();

                                var xend = xposlist[2] + 2*fntwd;
                                if ((coordedged.Width-2) < xend)
                                { xend = coordedged.Width - 3; }

                                var x1 = coordedged.SubMat(xupper, xbotm, xposlist[0], xposlist[1]);
                                var x2 = coordedged.SubMat(xupper, xbotm, xposlist[1], xposlist[2]);
                                var x3 = coordedged.SubMat(xupper, xbotm, xposlist[2], xposlist[2]+fntwd);
                                var x4 = coordedged.SubMat(xupper, xbotm, xposlist[2] + fntwd+1, xend);

                                var y1 = coordedged.SubMat(yupper, ybotm, xposlist[0], xposlist[1]);
                                var y2 = coordedged.SubMat(yupper, ybotm, xposlist[1], xposlist[2]);
                                var y3 = coordedged.SubMat(yupper, ybotm, xposlist[2], xposlist[2]+fntwd);
                                var y4 = coordedged.SubMat(yupper, ybotm, xposlist[2] + fntwd+1, xend);

                                cmlist.Add(x1); cmlist.Add(x2); cmlist.Add(x3); cmlist.Add(x4);
                                cmlist.Add(y1); cmlist.Add(y2); cmlist.Add(y3); cmlist.Add(y4);
                            }
                            else if (xposlist.Count == 2)
                            {
                                var fntlist = new List<int>();
                                fntlist.Add(xposlist[1] - xposlist[0]);
                                var fntwd = (int)fntlist.Average() + 1;
                                var xend = xposlist[1] + 3 * fntwd;
                                if ((coordedged.Width - 2) < xend)
                                { xend = coordedged.Width - 3; }

                                var x1 = coordedged.SubMat(xupper, xbotm, xposlist[0], xposlist[1]);
                                var x2 = coordedged.SubMat(xupper, xbotm, xposlist[1], xposlist[1] + fntwd);
                                var x3 = coordedged.SubMat(xupper, xbotm, xposlist[1] + fntwd + 1 , xposlist[1] + 2*fntwd);
                                var x4 = coordedged.SubMat(xupper, xbotm, xposlist[1] + 2 * fntwd + 1, xend);

                                var y1 = coordedged.SubMat(yupper, ybotm, xposlist[0], xposlist[1]);
                                var y2 = coordedged.SubMat(yupper, ybotm, xposlist[1], xposlist[1] + fntwd);
                                var y3 = coordedged.SubMat(yupper, ybotm, xposlist[1] + fntwd + 1, xposlist[1] + 2 * fntwd);
                                var y4 = coordedged.SubMat(yupper, ybotm, xposlist[1] + 2 * fntwd + 1, xend);

                                cmlist.Add(x1); cmlist.Add(x2); cmlist.Add(x3); cmlist.Add(x4);
                                cmlist.Add(y1); cmlist.Add(y2); cmlist.Add(y3); cmlist.Add(y4);
                            }
                            else
                            {
                                var xc = xposlist[0];
                                var xstep = (int)((coordedged.Width - xc) * 0.21);
                                var x1 = coordedged.SubMat(xupper, xbotm, xc, xstep + xc);
                                var x2 = coordedged.SubMat(xupper, xbotm, xstep + xc, 2 * xstep + xc);
                                var x3 = coordedged.SubMat(xupper, xbotm, 2 * xstep + xc, 3 * xstep + xc);
                                var x4 = coordedged.SubMat(xupper, xbotm, 3 * xstep + xc, 4 * xstep + xc);

                                var y1 = coordedged.SubMat(yupper, ybotm, xc, xstep + xc);
                                var y2 = coordedged.SubMat(yupper, ybotm, xstep + xc, 2 * xstep + xc);
                                var y3 = coordedged.SubMat(yupper, ybotm, 2 * xstep + xc, 3 * xstep + xc);
                                var y4 = coordedged.SubMat(yupper, ybotm, 3 * xstep + xc, 4 * xstep + xc);

                                cmlist.Add(x1); cmlist.Add(x2); cmlist.Add(x3); cmlist.Add(x4);
                                cmlist.Add(y1); cmlist.Add(y2); cmlist.Add(y3); cmlist.Add(y4);
                            }
                        }
                        else
                        {
                            var xc = 25;
                            var xstep = (int)((coordedged.Width - xc) * 0.21);
                            var x1 = coordedged.SubMat(xupper, xbotm, xc, xstep + xc);
                            var x2 = coordedged.SubMat(xupper, xbotm, xstep + xc, 2*xstep + xc);
                            var x3 = coordedged.SubMat(xupper, xbotm, 2 * xstep + xc, 3 * xstep + xc);
                            var x4 = coordedged.SubMat(xupper, xbotm, 3 * xstep + xc, 4 * xstep + xc);

                            var y1 = coordedged.SubMat(yupper, ybotm, xc, xstep + xc);
                            var y2 = coordedged.SubMat(yupper, ybotm, xstep + xc, 2 * xstep + xc);
                            var y3 = coordedged.SubMat(yupper, ybotm, 2 * xstep + xc, 3 * xstep + xc);
                            var y4 = coordedged.SubMat(yupper, ybotm, 3 * xstep + xc, 4 * xstep + xc);

                            cmlist.Add(x1); cmlist.Add(x2); cmlist.Add(x3); cmlist.Add(x4);
                            cmlist.Add(y1); cmlist.Add(y2); cmlist.Add(y3); cmlist.Add(y4);
                        }

                        //var idx = 1;
                        //foreach (var cm in cmlist)
                        //{
                        //    using (new Window("char "+idx, cm))
                        //    {
                        //        Cv2.WaitKey();
                        //    }
                        //    idx++;
                        //}

                    }//get coord box

                }//vline,hline
                
            }

        }

        public int CheckMatBlack(Mat edge, int startx, int endx, int upper, int botm)
        {
            var times = 0;
            for (var idx = startx; idx < endx;)
            {
                var snapmat = edge.SubMat(upper + 2, botm - 2, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2 && times >= 5)
                { return idx + 2; }
                idx = idx + 2;
                times++;
            }

            return -1;
        }

        public int GetXPos(Mat edge, int startx, int endx, int xupper, int xbotm, int yupper, int ybotm)
        {
            var xstart = CheckMatBlack(edge, startx, endx, xupper, xbotm);
            var ystart = CheckMatBlack(edge, startx, endx, yupper, ybotm);
            if (xstart != -1 && ystart != -1)
            {
                if (xstart > ystart)
                { return ystart; }
                else
                { return xstart; }
            }
            else if (xstart != -1)
            { return xstart; }
            else if (ystart != -1)
            { return ystart; }
            else
            { return -1; }
        }


        public int CheckMatWhite(Mat edge, int startx, int endx, int upper, int botm)
        {
            var times = 0;
            for (var idx = startx; idx < endx;)
            {
                var snapmat = edge.SubMat(upper+2, botm-2, idx, idx+2);
                var cnt = snapmat.CountNonZero();
                if (cnt >= 10 && times >= 3)
                { return idx - 2; }
                idx = idx + 2;
                times++;
            }

            return -1;
        }

        public int GetFirstPos(Mat edge, int startx, int endx, int xupper, int xbotm, int yupper, int ybotm)
        {
            var xstart = CheckMatWhite(edge, startx, endx, xupper, xbotm);
            var ystart = CheckMatWhite(edge, startx, endx, yupper, ybotm);
            if (xstart != -1 && ystart != -1)
            {
                if (xstart > ystart)
                { return xstart; }
                else
                { return ystart; }
            }
            else if (xstart != -1)
            { return xstart; }
            else if (ystart != -1)
            { return ystart; }
            else
            { return -1; }
        }

        public int GetCoordUpper(Mat edge,int xl,int xh,int midy,int miny)
        {
            for (var idx = midy; idx > miny;)
            {
                var snapmat = edge.SubMat(idx-2, idx, xl, xh);
                var cnt = snapmat.CountNonZero();
                if (cnt < 3)
                { return idx - 2; }
                idx = idx - 3;
            }

            return miny;
        }

        public int GetCoordBotm(Mat edge, int xl, int xh, int midy,int maxy)
        {
            for (var idx = midy; idx < maxy;)
            {
                var snapmat = edge.SubMat(idx, idx+2, xl, xh);
                var cnt = snapmat.CountNonZero();
                if (cnt < 3)
                { return idx+2; }
                idx = idx + 3;
            }

            return maxy;
        }






        public void RunXProc(string imgpath)
        {
            Mat srcimg = Cv2.ImRead(imgpath, ImreadModes.Color);

            var srcgray = new Mat();
            Cv2.CvtColor(srcimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(srcgray, blurred, new Size(3, 3), 0);

            //var edged = new Mat();
            //Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 6);

            var edged = new Mat();
            Cv2.Canny(blurred, edged, 50, 200, 3, true);

            var edgedx = new Mat(new Size(edged.Width, edged.Height), MatType.CV_32FC1);
            edged.ConvertTo(edgedx, MatType.CV_32FC1);

            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            var rectlist = new Rect[] { };
            var boxdetc = OpenCvSharp.XImgProc.EdgeBoxes.Create();
            boxdetc.GetBoundingBoxes(edgedx, edgedx, out rectlist);
        }

        public void line(string imgpath)
        {
            Mat srcimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var srcgray = new Mat();
            Cv2.CvtColor(srcimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var linedetc = OpenCvSharp.XImgProc.FastLineDetector.Create(180);
            var lines = linedetc.Detect(srcgray);
            foreach (var ln in lines)
            {
                Cv2.Line(srcimg, (int)ln.Item0, (int)ln.Item1, (int)ln.Item2, (int)ln.Item3, new Scalar(0, 255, 0), 3);
                using (new Window("srcimg1", srcimg))
                {
                    Cv2.WaitKey();
                }
            }

        }

        public void Histogram(string imgpath)
        {
            Mat srcrealimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            //var srcgray = new Mat();
            //Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);


            //using (new Window("srcimg1", srcrealimg))
            //{
            //    Cv2.WaitKey();
            //}

            const int Width = 260, Height = 200;
            Mat render = new Mat(new Size(Width, Height), MatType.CV_8UC3, Scalar.All(255));

            // Calculate histogram
            Mat hist = new Mat();
            int[] hdims = { 256 }; // Histogram size for each dimension
            Rangef[] ranges = { new Rangef(0, 256), }; // min/max 
            Cv2.CalcHist(
                new Mat[] { srcrealimg },
                new int[] { 0 },
                null,
                hist,
                1,
                hdims,
                ranges);

            // Get the max value of histogram
            double minVal, maxVal;
            Cv2.MinMaxLoc(hist, out minVal, out maxVal);

            Scalar color = Scalar.All(100);
            // Scales and draws histogram
            hist = hist * (maxVal != 0 ? Height / maxVal : 0.0);
            for (int j = 0; j < hdims[0]; ++j)
            {
                int binW = (int)((double)Width / hdims[0]);
                render.Rectangle(
                    new Point(j * binW, render.Rows - (int)(hist.Get<float>(j))),
                    new Point((j + 1) * binW, render.Rows),
                    color,
                    -1);
            }

            using (new Window("Histogram", WindowMode.AutoSize | WindowMode.FreeRatio, render))
            {
                Cv2.WaitKey();
            }

        }

        public void RUN_PD(string imgpath)
        {
            Mat srcrealimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            //var detectsize = GetDetectPoint(srcorgimg);
            //var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            //var angle = GetAngle(imgpath);

            //if (angle >= 0.7 && angle <= 359.3)
            //{
            //    var center = new Point2f(srcrealimg.Width / 2, srcrealimg.Height / 2);
            //    var m = Cv2.GetRotationMatrix2D(center, angle, 1);
            //    var outxymat = new Mat();
            //    Cv2.WarpAffine(srcrealimg, outxymat, m, new Size(srcrealimg.Width, srcrealimg.Height));
            //    srcrealimg = outxymat;
            //}

            //var srcenhance = new Mat();
            //Cv2.DetailEnhance(srcrealimg, srcenhance);

            //var srcgray = new Mat();
            //var denoisemat2 = new Mat();
            //Cv2.FastNlMeansDenoisingColored(srcenhance, denoisemat2, 10, 10, 7, 21);

            //srcrealimg = denoisemat2;

            using (new Window("srcimg1", srcrealimg))
            {
                Cv2.WaitKey();
            }

            var coordmat = srcrealimg.SubMat((int)(srcrealimg.Height * 0.2), (int)(srcrealimg.Height * 0.4), (int)(srcrealimg.Width * 0.20), (int)(srcrealimg.Width * 0.8));
            using (new Window("coordmat", coordmat))
            {
                Cv2.WaitKey();
            }


            var xyenhance = new Mat();

            Cv2.DetailEnhance(coordmat, xyenhance);
            var denoisemat1 = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance, denoisemat1, 10, 10, 7, 21);
            xyenhance = denoisemat1;

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 2, xyenhance.Height * 2));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);

            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            using (new Window("xyenhance4x", xyenhance4x))
            {
                Cv2.WaitKey();
            }

            using (new Window("denoisemat", denoisemat))
            {
                Cv2.WaitKey();
            }

            using (new Window("xyenhgray", xyenhgray))
            {
                Cv2.WaitKey();
            }

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(3, 3), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9,6);

            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

        }

        public void RUN_IIVI(string imgpath,OpenCvSharp.ML.ANN_MLP kmode=null)
        {
            //img height/circle radius  3.1 ~ 4.2

            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());
            //using (new Window("srcimg1", srcrealimg))
            //{
            //    Cv2.WaitKey();
            //}

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 80, 115, 140);

            if (circles.Count() > 0)
            {
                var ccl = circles[0];
                //Cv2.Circle(srcrealimg, (int)ccl.Center.X, (int)ccl.Center.Y, (int)ccl.Radius, new Scalar(0, 255, 0), 3);
                //using (new Window("srcimg2", srcrealimg))
                //{
                //    Cv2.WaitKey();
                //}

                var halfheight = srcrealimg.Height / 2;
                if (ccl.Center.Y < halfheight)
                {
                    var outxymat = new Mat();
                    Cv2.Transpose(srcrealimg, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    Cv2.Transpose(outxymat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    srcrealimg = outxymat;

                    srcgray = new Mat();
                    Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);
                    circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 80, 115, 140);
                    ccl = circles[0];
                }

                var rat = srcrealimg.Height / ccl.Radius;
                //using (new Window("srcgray", srcgray))
                //{
                //    Cv2.WaitKey();
                //}

                var xcoordx = (int)(ccl.Center.X + 100);
                var xcoordy = (int)(ccl.Center.Y - 53);
                var ycoordx = (int)(ccl.Center.X + 7);
                var ycoordy = (int)(ccl.Center.Y - 259);

                var markx = (int)(ccl.Center.X + 123);
                var marky = (int)(ccl.Center.Y - 125);

                if (ycoordy < 0) { ycoordy = 3; }

                var ximg = srcrealimg.SubMat(new Rect(xcoordx,xcoordy,90,54));
                var yimg = srcrealimg.SubMat(new Rect(ycoordx,ycoordy,54,90));
                {
                    var outxymat = new Mat();
                    Cv2.Transpose(yimg, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    yimg = outxymat;
                }

                var combinimg = new Mat();
                Cv2.HConcat(ximg, yimg, combinimg);

                var markgrey = srcgray.SubMat(new Rect(markx,marky,60,60));
                var markmat = new Mat();
                Cv2.AdaptiveThreshold(markgrey, markmat, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);


                ximg = GetEnhanceEdge(ximg);
                yimg = GetEnhanceEdge(yimg);

                var charlist = new List<Mat>();
                charlist.Add(combinimg);
                charlist.Add(markmat);
                charlist.AddRange(GetCharMats(ximg,1));
                charlist.Add(markmat);
                charlist.AddRange(GetCharMats(yimg,2));

                var idx = 0;
                foreach (var cm in charlist)
                {
                    if (idx == 0)
                    {
                        //using (new Window("cmxxxxxxxxxxxxx" + idx, cm))
                        //{
                        //    Cv2.WaitKey();
                        //}
                    }
                    else
                    {
                        var tcm = new Mat();
                        cm.ConvertTo(tcm, MatType.CV_32FC1);
                        var tcmresize = new Mat();
                        Cv2.Resize(tcm, tcmresize, new Size(30, 30), 0, 0, InterpolationFlags.Linear);
                        using (new Window("cmxxxxxxxxxxxxx"+idx, tcmresize))
                        {
                            Cv2.WaitKey();
                        }

                        if (kmode != null)
                        {
                            var resultmat = new Mat();
                            var stcm = tcmresize.Reshape(0, 1);

                            var matched = new Mat();
                            var m = kmode.Predict(stcm, matched);//,OpenCvSharp.ML.StatModel.Flags.RawOutput);
                                                                 //kmode.FindNearest(stcm, 7, resultmat, matched);
                            var matchstr = matched.Dump();
                            var ms = matchstr.Split(new string[] { "[", "]", "," }, StringSplitOptions.RemoveEmptyEntries);
                            var mstr = "";
                            foreach (var s in ms)
                            {
                                //mstr += UT.O2S((char)UT.O2I(s));
                                mstr += UT.O2S(UT.O2D(s));
                            }

                            var blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
                            Cv2.PutText(blank, m.ToString(), new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                            using (new Window("blank", blank))
                            {
                                Cv2.WaitKey();
                            }
                        }

                    }
                    idx++;
                }
            }
            else
            {

            }

        }


        public List<Mat> GetCharMats(Mat xymat,int id)
        {
            var charlist = new List<Mat>();

            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(xymat, null, out kazeKeyPoints, kazeDescriptors);

            var wptlist = new List<KeyPoint>();
            for (var idx = 20; idx < xymat.Width;)
            {
                var yhlist = new List<double>();
                var wlist = new List<KeyPoint>();
                foreach (var pt in kazeKeyPoints)
                {
                    if (pt.Pt.X >= (idx - 20) && pt.Pt.X < idx)
                    {
                        if (pt.Pt.Y + 6 > xymat.Height)
                        { continue; }

                        wlist.Add(pt);
                        yhlist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 10 && (yhlist.Max() - yhlist.Min()) > 0.3 * xymat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 20;
            }

            var xlist = new List<double>();
            var ylist = new List<double>();
            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
                ylist.Add(pt.Pt.Y);
            }

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(xymat, wptlist, dstKaze);
            using (new Window("dstKazexx"+id, dstKaze))
            {
                Cv2.WaitKey();
            }

            var h0 = (int)ylist.Min() - 3;
            if (h0 < 0) { h0 = 0; }
            var h1 = (int)ylist.Max() + 3;
            if (h1 - h0 > 158) { h1 = h0 + 158; }
            if (h1 > xymat.Height) { h1 = xymat.Height - 1; }

            var xmax = (int)xlist.Max();

            var start = xmax - 150; var end = xmax - 30;
            var split2x = GetSplitX(xymat, start, end,h0,h1);

            start = xmax - 140; end = (xmax - 250) < 0 ? 0 : (xmax - 250);
            if (split2x != -1)
            { start = split2x - 150; end = split2x - 40; }

            var split1x = GetSplitX(xymat,start, end,h0,h1);

            start = xmax - 260; end = (xmax - 380) < 0 ? 0 : (xmax - 380);
            if (split1x != -1)
            { start = split1x - 150; end = split1x - 40; }

            var split0x = GetSplitX(xymat, start, end, h0, h1);

            if (split1x != -1 && split2x != -1 && split0x != -1)
            {
                charlist.Add(xymat.SubMat(h0, h1, split0x, split1x));
                charlist.Add(xymat.SubMat(h0, h1, split1x, split2x));
                charlist.Add(xymat.SubMat(h0, h1, split2x, xymat.Width - 3));
            }
            else if (split1x != -1 && split2x != -1)
            {
                var fontwd = split2x - split1x;
                var x0 = split1x - fontwd;
                if (x0 < 0) { x0 = 0; }

                charlist.Add(xymat.SubMat(h0, h1, x0, split1x));
                charlist.Add(xymat.SubMat(h0, h1, split1x, split2x));
                charlist.Add(xymat.SubMat(h0, h1, split2x, xymat.Width - 3));
            }
            else if (split2x != -1)
            {
                var fontwd = xymat.Width - split2x;
                var x1 = split2x - fontwd;
                var x0 = split2x - 2*fontwd;
                if (x0 < 0) { x0 = 0; }

                charlist.Add(xymat.SubMat(h0, h1, x0, x1));
                charlist.Add(xymat.SubMat(h0, h1, x1, split2x));
                charlist.Add(xymat.SubMat(h0, h1, split2x, xymat.Width - 3));
            }
            else
            {
                var x0 = xymat.Width - 304;
                if (x0 < 0) { x0 = 0; }
                charlist.Add(xymat.SubMat(h0, h1,x0 , xymat.Width - 204));
                charlist.Add(xymat.SubMat(h0, h1, xymat.Width - 204, xymat.Width - 103));
                charlist.Add(xymat.SubMat(h0, h1, xymat.Width - 103, xymat.Width-3));
            }

            return charlist;

        }

        public int GetSplitX(Mat xymat, int snapstart, int snapend, int h0, int h1)
        {
            var ret = -1;
            var tm = 0;
            for (var sidx = snapend; sidx > snapstart;)
            {
                var snapmat = xymat.SubMat(h0, h1, sidx, sidx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 3)
                {
                    if (ret != -1 && tm == 1)
                    { return sidx; }
                    else
                    { ret = sidx; tm = 1; }
                }
                else
                { tm = 0; }
                sidx = sidx - 2;
            }
            return -1;
        }

        public int GetSplitX1(Mat xymat, int snapstart, int snapend,int h0,int h1)
        {
            bool hassplit = false;
            var wtob = 0;
            var btow = 0;
            var previouscolor = 1;

            for (var sidx = snapend; sidx > snapstart; )
            {
                var snapmat = xymat.SubMat(h0, h1, sidx, sidx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    hassplit = true;
                    if (previouscolor == 1)
                    {
                        previouscolor = 0;
                        wtob = sidx;
                    }
                    previouscolor = 0;
                }
                else
                {
                    if (previouscolor == 0)
                    {
                        btow = sidx;
                        break;
                    }
                    previouscolor = 1;
                }

                sidx = sidx - 2;
            }

            if (hassplit && wtob != 0 && btow != 0)
            { return (wtob + btow) / 2; }

            return 0;
        }

        public Mat GetEnhanceEdge(Mat xymat)
        {
            var xyenhance4x = new Mat();
            Cv2.Resize(xymat, xyenhance4x, new Size(xymat.Width * 4, xymat.Height * 4));
           Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(3, 3), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            //using (new Window("edged", edged))
            //{
            //    Cv2.WaitKey();
            //}

            return edged;
        }


        public  void DetectSplitLine(string imgpath)
        {
            var angle = GetAngle(imgpath);

            var xyrectlist = FindXYRect5X1(imgpath, angle, 25, 43, 4800, 8000);
            if (xyrectlist.Count > 0)
            {
                Mat src = Cv2.ImRead(imgpath, ImreadModes.Color);
                if (angle >= 0.7 && angle <= 359.3)
                {
                    var center = new Point2f(src.Width / 2, src.Height / 2);
                    var m = Cv2.GetRotationMatrix2D(center, angle, 1);
                    var outxymat = new Mat();
                    Cv2.WarpAffine(src, outxymat, m, new Size(src.Width, src.Height));
                    src = outxymat;
                }

                var xyrect = xyrectlist[0];
                var xymat = src.SubMat(xyrect);

                var availableimgpt = GetDetectPoint(src);
                //var srcmidy = src.Height / 2;
                var srcmidy = (availableimgpt[1].Max() + availableimgpt[1].Min()) / 2;

                if (xyrect.Y + xyrect.Height > srcmidy)
                {
                    var center = new Point2f(xymat.Width / 2, xymat.Height / 2);
                    var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                    var outxymat = new Mat();
                    Cv2.WarpAffine(xymat, outxymat, m, new Size(xymat.Width, xymat.Height));
                    xymat = outxymat;
                }


                var xyenhance = new Mat();
                Cv2.DetailEnhance(xymat, xyenhance);

                var xyenhance4x = new Mat();
                Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));

                Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

                var xyenhgray = new Mat();
                var denoisemat = new Mat();
                Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
                Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);


                var blurred = new Mat();
                Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

                var edged = new Mat();
                Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

                using (new Window("edged", edged))
                {
                    Cv2.WaitKey();
                }

                var orgxlist = new List<int>();

                var detect_h0 = (int)(edged.Height * 0.25);
                var detect_h1 = (int)(edged.Height * 0.75);

                var detect_h0x = (int)(edged.Height * 0.33);
                var detect_h1x = (int)(edged.Height * 0.67);
                var edgestart = 50;
                var edgeend = edged.Width - 50;



                var width_bond = edged.Width - 15;
                var preidx_color = 1;
                var preidx_mark = 0;

                for (var crt_ptx = 10; crt_ptx < width_bond;)
                {
                    var line = new Mat();
                    if (crt_ptx < edgestart || crt_ptx > edgeend)
                    {
                        line = edged.SubMat(detect_h0x, detect_h1x, crt_ptx, crt_ptx + 3);
                    }
                    else
                    {
                       line = edged.SubMat(detect_h0, detect_h1, crt_ptx, crt_ptx + 3);
                    }

                    //using (new Window("line", line))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    var cnt = line.CountNonZero();
                    if (cnt <= 2)
                    {
                        if (preidx_color == 1)
                        {
                            if (orgxlist.Count > 0)
                            {
                                var lastidx = orgxlist[orgxlist.Count - 1];
                                var seg = crt_ptx - lastidx;
                                if (seg >= 30 && seg <= 70)
                                {
                                    orgxlist.Add(crt_ptx);
                                }
                            }
                        }

                        preidx_color = 0;
                        preidx_mark = crt_ptx;
                    }

                    if (cnt > 2)
                    {
                        if (preidx_color == 0)
                        {
                            if (orgxlist.Count > 0)
                            {
                                var lastidx = orgxlist[orgxlist.Count - 1];
                                var seg = preidx_mark - lastidx;
                                if (seg > 15 && seg < 30)
                                {
                                    var x = 0;
                                }
                                else
                                { orgxlist.Add(preidx_mark); }
                            }
                            else
                            { orgxlist.Add(preidx_mark); }
                        }

                        preidx_color = 1;
                        preidx_mark = crt_ptx;
                    }

                    crt_ptx = crt_ptx + 3;
                }//end for

                var cut_eh0 = 15;
                var cut_eh1 = edged.Height - 15;

                for (var idx = 1; idx < orgxlist.Count; idx++)
                {
                    var wd = orgxlist[idx] - orgxlist[idx - 1];
                    if ( wd > 35 && wd < 70)
                    {
                        var sub = edged.SubMat(cut_eh0, cut_eh1, orgxlist[idx - 1], orgxlist[idx]);
                        using (new Window("sub"+idx, sub))
                        {
                            Cv2.WaitKey();
                        }
                    }
                }//end for

            }//end if
        }

        public static void removehoriz(string imgpath)
        {
            Mat gray = Cv2.ImRead(imgpath, ImreadModes.Grayscale);


                var bw = new Mat();
                Cv2.AdaptiveThreshold(~gray, bw, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 15, -2);

                using (new Window("bw", bw))
                {
                    Cv2.WaitKey();
                }

                //var horiz = bw.Clone();
            var vert = bw.Clone();

                //var horiz_size = horiz.Cols / 30;
                //var horizstruct = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(horiz_size, 1));
                //Cv2.Erode(horiz, horiz, horizstruct, new Point(-1, -1));

                //using (new Window("horiz1", horiz))
                //{
                //    Cv2.WaitKey();
                //}

                //Cv2.Dilate(horiz, horiz, horizstruct, new Point(-1, -1));

                //using (new Window("horiz2", horiz))
                //{
                //    Cv2.WaitKey();
                //}

            var vert_size = vert.Rows / 30;
            var vertstruct = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, vert_size));

            Cv2.Erode(vert, vert, vertstruct, new Point(-1, -1));

            using (new Window("vert1", vert))
            {
                Cv2.WaitKey();
            }

            Cv2.Dilate(vert, vert, vertstruct, new Point(-1, -1));
            using (new Window("vert2", vert))
            {
                Cv2.WaitKey();
            }

            Cv2.BitwiseNot(vert, vert);

                using (new Window("vert3", vert))
                {
                    Cv2.WaitKey();
                }

                var eg = new Mat();
                Cv2.AdaptiveThreshold(vert, eg, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 3, -2);


                Mat kernel = Mat.Ones(2, 2, MatType.CV_8UC1);
                Cv2.Dilate(eg, eg, kernel);

                using (new Window("eg", eg))
                {
                    Cv2.WaitKey();
                }

                var smooth = new Mat();
                vert.CopyTo(smooth);
                Cv2.Blur(smooth, smooth, new Size(2, 2));
                smooth.CopyTo(vert, eg);

                using (new Window("g2", vert))
                {
                    Cv2.WaitKey();
                }

        }


        public static void GetCaptureImg(string imgpath)
        {
            var waferlist = new List<string>(new string[] { "192926-20E", "193013-10E", "192830-10E", "192824-10E", "192823-80E", "191726-20E" });

            foreach (var wf in waferlist)
            {
                var sql = "SELECT TOP 4 [CaptureImg],wafernum+appv_3 FROM [WAT].[dbo].[OGPFatherImg] where wafernum = @wafer";
                var dict = new Dictionary<string, string>();
                dict.Add("@wafer", wf);
                var dbret = DBUtility.ExeLocalSqlWithRes(sql, dict);
                foreach (var line in dbret)
                {
                    var imgstr = UT.O2S(line[0]);
                    var f = UT.O2S(line[1]);
                    WriteImg(imgpath, f, imgstr);
                }
            }
        }

        public static void WriteImg(string imgpath, string fn, string imgdata)
        {
            var newfn = imgpath + fn + ".png";
            var bytes = Convert.FromBase64String(imgdata);
            var src = Cv2.ImDecode(bytes, ImreadModes.Color);

            var enh = new Mat();
            Cv2.DetailEnhance(src, enh);

            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(enh, denoisemat, 10, 10, 7, 21);

            var grey = new Mat();
            Cv2.CvtColor(denoisemat, grey, ColorConversionCodes.BGR2GRAY);

            var blur = new Mat();
            Cv2.GaussianBlur(grey, blur, new Size(1, 1), 0);

            Mat des = new Mat();
            Cv2.Threshold(blur, des, 160, 255, ThresholdTypes.BinaryInv);
            //Cv2.AdaptiveThreshold(~grey, des, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 15, -2);

            Mat des2 = new Mat();
            Cv2.Resize(des, des2, new Size(des.Width * 2, des.Height * 2));
            Cv2.ImWrite(newfn, des2);
        }


        public static List<OpenCvSharp.Rect> GetSmall5x1Rect(Mat blurred, Mat srcgray,Mat srcenhance,bool cflag)
        {
            var ret = new List<OpenCvSharp.Rect>();
            //var cflaglist = new List<bool>();
            //cflaglist.Add(false);
            //cflaglist.Add(true);
            //foreach(var cflag in cflaglist)
            //{s
                var edged = new Mat();
                Cv2.Canny(blurred, edged, 50, 200, 3, cflag);

                using (new Window("edged", edged))
                {
                    Cv2.WaitKey();
                }

                var outmat = new Mat();
                var ids = OutputArray.Create(outmat);
                var cons = new Mat[] { };
                Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
                var conslist = cons.ToList();

                var idx = 0;
                foreach (var item in conslist)
                {
                    idx++;

                    var rect = Cv2.BoundingRect(item);
                    var whrate = (double)rect.Height / (double)rect.Width;
                    var a = rect.Width * rect.Height;
                    if (rect.Width >= 18 && rect.Width <= 35
                        && whrate >= 4.5 && whrate < 8 && a < 5500 && rect.Y > 50)
                    {
                    var xymat = srcgray.SubMat(rect);
                    //using (new Window("xymat" + idx, xymat))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    if ((rect.Width >= 18 && rect.Width <= 20) && ((rect.X+rect.Width+2) <= edged.Width))
                        {
                            rect = new OpenCvSharp.Rect(rect.X - 2, rect.Y, rect.Width + 4, rect.Height);
                        }

                        ret.Add(rect);
                        //if (ret.Count > 0)
                        //{
                        //    if (a > ret[0].Width * ret[0].Height)
                        //    {
                        //        ret.Clear();
                        //        ret.Add(rect);
                        //    }
                        //}
                        //else
                        //{ ret.Add(rect); }
                    }
                }

                if (ret.Count > 0)
                {
                    var crect = ret[0];
                    if (ret.Count > 1)
                    {
                        ret.Sort(delegate (OpenCvSharp.Rect obj1,OpenCvSharp.Rect obj2)
                        {
                            var a1 = obj1.Width * obj1.Height;
                            var a2 = obj2.Width * obj2.Height;
                            return a1.CompareTo(a2);
                        });
                        crect = ret[1];
                    }

                    var xymat = srcenhance.SubMat(crect);
                    var outxymat = new Mat();
                    Cv2.Transpose(xymat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    Cv2.Resize(outxymat, outxymat, new Size(outxymat.Width * 4, outxymat.Height * 4));
                    Cv2.DetailEnhance(outxymat, outxymat);
                    var xlist = GetCoordWidthPT_W(outxymat, outxymat);
                    if (xlist.Count > 0)
                    {
                        var xmax = xlist.Max();
                        var xmin = xlist.Min();
                        if (xmax - xmin >= 400)
                        {
                            ret = new List<OpenCvSharp.Rect>();
                            ret.Add(crect);
                            return ret;
                        }
                        else
                        { ret.Clear(); }
                    }
                }

                //foreach (var item in conslist)
                //{
                //    idx++;

                //    var rect = Cv2.BoundingRect(item);
                //    var whrate =  (double)rect.Width /(double)rect.Height ;
                //    var a = rect.Width * rect.Height;
                //    if (rect.Height >= 18 && rect.Height <= 34
                //        && whrate >= 4.5 && whrate < 8 && a < 5000 && rect.X > 50)
                //    {
                //        var xymat = srcgray.SubMat(rect);
                //        using (new Window("xymat" + idx, xymat))
                //        {
                //            Cv2.WaitKey();
                //        }

                //        if ((rect.Height >= 18 && rect.Height <= 20) && ((rect.Y + rect.Height + 2) <= edged.Height))
                //        { rect = new OpenCvSharp.Rect(rect.X, rect.Y-2, rect.Width, rect.Height+4); }

                //        if (ret.Count > 0)
                //        {
                //            if (a > ret[0].Width * ret[0].Height)
                //            {
                //                ret.Clear();
                //                ret.Add(rect);
                //            }
                //        }
                //        else
                //        { ret.Add(rect); }
                //    }
                //}

                //if (ret.Count > 0)
                //{ return ret; }
            //}

            return new List<OpenCvSharp.Rect>();
        }



        public void Runsm5x1(string imgpath)
        {
            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            var srcenhance = new Mat();
            Cv2.DetailEnhance(srcrealimg, srcenhance);

            var srcgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(srcenhance, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, srcgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(srcgray, blurred, new Size(3, 3), 0);

            var frects = GetSmall5x1Rect(blurred, srcgray,srcenhance,false);
            var trects = GetSmall5x1Rect(blurred, srcgray, srcenhance, true);

            var rects = new List<OpenCvSharp.Rect>();
            if (frects.Count > 0 && trects.Count > 0)
            {
                if (frects[0].Width * frects[0].Height < trects[0].Width * trects[0].Height)
                { rects.AddRange(frects); }
                else
                { rects.AddRange(trects); }
            }
            else if (frects.Count > 0)
            { rects.AddRange(frects); }
            else if (trects.Count > 0)
            { rects.AddRange(trects); }

            if (rects.Count > 0)
            {
                var coormat = srcenhance.SubMat(rects[0]);
                //using (new Window("coormat", coormat))
                //{
                //    Cv2.WaitKey();
                //}

                if (rects[0].Height > rects[0].Width)
                {
                    //if (coormat.Width % 2 == 1)
                    //{ w = coormat.Width + 1; }
                    //var center = new Point2f(0,0);
                    //var m = Cv2.GetRotationMatrix2D(center, 90, 1);
                    var outxymat = new Mat();
                    //Cv2.WarpAffine(coormat, outxymat, m,new Size(coormat.Rows,coormat.Rows));
                    Cv2.Transpose(coormat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    Cv2.Transpose(outxymat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    Cv2.Transpose(outxymat, outxymat);
                    Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                    coormat = outxymat;
                }

                //Cv2.DetailEnhance(coormat, coormat);
                //using (new Window("coormat", coormat))
                //{
                //    Cv2.WaitKey();
                //}

                var coormatresize = new Mat();
                Cv2.Resize(coormat, coormatresize, new Size(coormat.Cols * 4, coormat.Rows * 4),0,0,InterpolationFlags.Linear);

                //{
                //    var charmat4x = coormatresize.Clone();
                //    var kaze = KAZE.Create();
                //    var kazeDescriptors = new Mat();
                //    KeyPoint[] kazeKeyPoints = null;
                //    kaze.DetectAndCompute(charmat4x, null, out kazeKeyPoints, kazeDescriptors);
                //    var hptlist = new List<KeyPoint>();
                //    var cl = 0.3 * charmat4x.Height;
                //    var ch = 0.7 * charmat4x.Height;
                //    var rl = 60;
                //    var rlh = charmat4x.Width * 0.3;
                //    var rhl = charmat4x.Width * 0.7;
                //    var rh = charmat4x.Width - 60;

                //    foreach (var pt in kazeKeyPoints)
                //    {
                //        if (pt.Pt.Y >= cl && pt.Pt.Y <= ch
                //            && ((pt.Pt.X >= rl && pt.Pt.X <= rlh) || (pt.Pt.X >= rhl && pt.Pt.X <= rh)))
                //        {
                //            hptlist.Add(pt);
                //        }
                //    }


                //    if (hptlist.Count < 100)
                //    {
                //        var dstKaze = new Mat();
                //        Cv2.DrawKeypoints(charmat4x, hptlist.ToArray(), dstKaze);
                //        using (new Window("less point dstKaze", dstKaze))
                //        {
                //            Cv2.WaitKey();
                //        }
                //    }
                //    else
                //    {
                //        var dstKaze = new Mat();
                //        Cv2.DrawKeypoints(charmat4x, hptlist.ToArray(), dstKaze);
                //        using (new Window("more point dstKaze", dstKaze))
                //        {
                //            Cv2.WaitKey();
                //        }
                //    }
                //}


                //using (new Window("coormatresize", coormatresize))
                //{
                //    Cv2.WaitKey();
                //}

                var coorenhance = new Mat();
                Cv2.DetailEnhance(coormatresize, coorenhance);
                //Cv2.DetailEnhance(coorenhance, coorenhance);

                using (new Window("coorenhance", coorenhance))
                {
                    Cv2.WaitKey();
                }

                //SRMat(coormat);

                var coorgray = new Mat();
                var denoisemat2 = new Mat();
                Cv2.FastNlMeansDenoisingColored(coorenhance, denoisemat2, 10, 10, 7, 21);
                Cv2.CvtColor(denoisemat2, coorgray, ColorConversionCodes.BGR2GRAY);
                Cv2.GaussianBlur(coorgray, blurred, new Size(13, 13), 2.6);

                using (new Window("coorgray", coorgray))
                {
                    Cv2.WaitKey();
                }

                var edged = new Mat();
                Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

                //var lowspec = new Scalar(0, 0, 0);
                ////var highspec = new Scalar(100, 67, 65);
                //var highspec = new Scalar(113, 102, 56);
                //var coordrgb = new Mat();
                //Cv2.CvtColor(coorenhance, coordrgb, ColorConversionCodes.BGR2RGB);
                //var edged = coordrgb.InRange(lowspec, highspec);

                using (new Window("edged12", edged))
                {
                    Cv2.WaitKey();
                }


                var rectlist = GetSmall5x1CharRect( blurred,  edged, coorenhance);

                var idx = 0;
                foreach (var rect in rectlist)
                {
                    var cmat = edged.SubMat(rect);
                    using (new Window("cmat"+idx, cmat))
                    {
                        Cv2.WaitKey();
                    }
                    idx++;
                }


                //var edged2 = new Mat();
                //Cv2.AdaptiveThreshold(blurred, edged2, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 11, 20);
                ////using (new Window("edged14", edged2))
                ////{
                ////    Cv2.WaitKey();
                ////}


                //var edged4 = new Mat();
                //Cv2.AdaptiveThreshold(blurred, edged4, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 23, 20);
                ////using (new Window("edged16", edged4))
                ////{
                ////    Cv2.WaitKey();
                ////}

                //var edged5 = new Mat();
                //Cv2.AdaptiveThreshold(blurred, edged5, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 11);


                //var struc = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(3, 3));
                //var erodemat = new Mat();
                //Cv2.Erode(edged, erodemat, struc);
                ////using (new Window("erodemat", erodemat))
                ////{
                ////    Cv2.WaitKey();
                ////}

                //var matlist = new List<Mat>();
                //matlist.Add(edged);
                ////matlist.Add(edged1);
                //matlist.Add(edged2);
                ////matlist.Add(edged3);
                //matlist.Add(edged4);
                //matlist.Add(edged5);
                //matlist.Add(erodemat);

                ////var coorsobel = new Mat();
                ////Cv2.Sobel(coorgray, coorsobel, MatType.CV_8U, 1, 0, 3, 1, 0, BorderTypes.Replicate);
                ////using (new Window("coorsobel", coorsobel))
                ////{
                ////    Cv2.WaitKey();
                ////}
                ////var coorthod = new Mat();
                ////Cv2.Threshold(coorsobel, coorthod, 0, 255, ThresholdTypes.Otsu|ThresholdTypes.Binary);
                ////using (new Window("coorthod1", coorthod))
                ////{
                ////    Cv2.WaitKey();
                ////}

                ////var dilatemat = new Mat();
                ////Cv2.Dilate(erodemat, dilatemat, struc);

                ////using (new Window("dilatemat", dilatemat))
                ////{
                ////    Cv2.WaitKey();
                ////}

                //foreach (var m in matlist)
                //{
                //    var outmat = new Mat();
                //    var ids = OutputArray.Create(outmat);
                //    var cons = new Mat[] { };
                //    Cv2.FindContours(m, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                //    var idx1 = 0;
                //    foreach (var item in cons)
                //    {
                //        idx1++;

                //        var crect = Cv2.BoundingRect(item);

                //        if (crect.Width > 36 && crect.Width <= 60 && crect.Height > 54 && crect.Height <= 65  && crect.Y > 8)
                //        {
                //            Cv2.Rectangle(coorenhance, crect, new Scalar(0, 255, 0));
                //            using (new Window("xyenhance4", coorenhance))
                //            {
                //                Cv2.WaitKey();
                //            }
                //        }
                //    }
                //}//end foreach


            }

        }

        public List<OpenCvSharp.Rect> GetSmall5x1CharRect(Mat blurred, Mat edged,Mat coorenhance)
        {

            var cbond = GetCoordHighPT(blurred, edged);

            //var coorenhance2 = new Mat();
            //Cv2.DetailEnhance(coorenhance, coorenhance2); 
            var xlist = GetCoordWidthPT(coorenhance);

            if (xlist.Count == 0 || (xlist.Max() - xlist.Min()) < 400)
            { return new List<OpenCvSharp.Rect>(); }

            var xmid = (xlist.Max() + xlist.Min()) / 2;
            var xcxlist = new List<double>();
            var ycxlist = new List<double>();
            foreach (var x in xlist)
            {
                if (x < xmid)
                { xcxlist.Add(x); }
                else
                { ycxlist.Add(x); }
            }

            var xwd = xcxlist.Max() - xcxlist.Min();
            var ywd = ycxlist.Max() - ycxlist.Min();
            var blankwidth = ycxlist.Min() - xcxlist.Max();

            var xcmax = xcxlist.Max() + 3;
            var ycmin = ycxlist.Min() - 3;
            if (Math.Abs(ywd - xwd) > 30 && blankwidth > 220)
            {
                if (ywd > xwd)
                { xcmax = xcxlist.Max() + 47; }
                else
                { ycmin = ycxlist.Min() - 47; }
            }

            if (Math.Abs((ycmin - xmid) - (xmid - xcmax)) >= 14)
            {
                if ((ycmin - xmid) > (xmid - xcmax))
                { ycmin = ycmin - 10; }
                else
                { xcmax = xcmax + 10; }
            }
            else if (Math.Abs((ycmin - xmid) - (xmid - xcmax)) >= 8)
            {
                if ((ycmin - xmid) > (xmid - xcmax))
                { ycmin = ycmin - 5; }
                else
                { xcmax = xcmax + 5; }
            }

            var ret = new List<OpenCvSharp.Rect>();

            if (cbond.Count > 0)
            {

                var y0list = new List<int>();
                var y1list = new List<int>();
                                
                cbond.Sort(delegate (OpenCvSharp.Rect o1, OpenCvSharp.Rect o2)
                { return o1.X.CompareTo(o2.X); });

                var filteredbond = new List<OpenCvSharp.Rect>();
                foreach (var item in cbond)
                {
                    if (item.X <= 3)
                    { continue; }

                    if (filteredbond.Count == 0)
                    {
                        filteredbond.Add(item);
                        y0list.Add(item.Y);
                        y1list.Add(item.Height);
                    }
                    else
                    {
                        var bcnt = filteredbond.Count;
                        if (item.X - filteredbond[bcnt - 1].X > 28)
                        {
                            filteredbond.Add(item);
                            y0list.Add(item.Y);
                            y1list.Add(item.Height);
                        }
                    }
                }

                var y0 = (int)y0list.Average();
                var y1 = y1list.Max();

                if ((int)xcmax - 179 > 0) {
                    ret.Add(new OpenCvSharp.Rect((int)xcmax - 179, y0, 44, y1));
                }
                else {
                    ret.Add(new OpenCvSharp.Rect(0, y0, 44, y1));
                }

                ret.Add(new OpenCvSharp.Rect((int)xcmax - 130, y0, 43, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 90, y0, 44, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 46, y0, 45, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin, y0, 45, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 52, y0, 45, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 90, y0, 45, y1));

                if (((int)ycmin + 135 + 46) >= (edged.Cols-1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 135, y0, edged.Cols - (int)ycmin - 135, y1));
                }
                else
                { ret.Add(new OpenCvSharp.Rect((int)ycmin + 135, y0, 46, y1)); }



                var changedict = new Dictionary<int, bool>();

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X > ret[idx].X - 15) && (item.X < ret[idx].X + 15))
                        {
                            var currentrect = new OpenCvSharp.Rect(item.X, ret[idx].Y, item.Width, ret[idx].Height);
                            if (!changedict.ContainsKey(idx))
                            {
                                ret[idx] = currentrect;
                                changedict.Add(idx, true);
                            }
                            break;
                        }
                    }
                }//end for

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X > ret[idx].X - 15) && (item.X < ret[idx].X + 15))
                        {
                            if ((idx >= 0 && idx <= 2) || (idx >= 4 && idx <= 6))
                            {
                                var nextrect = new OpenCvSharp.Rect(item.X + item.Width + 1, ret[idx].Y
                                    , ((item.X + 2 * item.Width + 1) < edged.Width) ? item.Width : (edged.Width - item.X - item.Width - 1), ret[idx].Height);

                                if (!changedict.ContainsKey(idx + 1))
                                {
                                    ret[idx + 1] = nextrect;
                                    changedict.Add(idx + 1, true);
                                }
                            }

                            if ((idx >= 1 && idx <= 3) || (idx >= 5 && idx <= 7))
                            {
                                var nextrect = new OpenCvSharp.Rect((item.X - item.Width - 1) > 0 ? (item.X - item.Width) : 0, ret[idx].Y, item.Width + 1, ret[idx].Height);
                                if (!changedict.ContainsKey(idx - 1))
                                {
                                    ret[idx - 1] = nextrect;
                                    changedict.Add(idx - 1, true);
                                }
                            }
                            break;
                        }
                    }
                }//end for

            }
            else
            {
                var y0 = 18;
                var y1 = 69;
                if (edged.Height < 88)
                { y1 = edged.Height - y0; }

                if ((int)xcmax - 179 > 0) { ret.Add(new OpenCvSharp.Rect((int)xcmax - 179, y0, 44, y1)); }
                else { ret.Add(new OpenCvSharp.Rect(0, y0, 44, y1));}
                
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 130, y0, 43, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 90, y0, 44, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 46, y0, 45, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin, y0, 47, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 52, y0, 45, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 90, y0, 45, y1));

                if (((int)ycmin + 135 + 46) >= (edged.Cols-1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 135, y0, edged.Cols - (int)ycmin - 135, y1));
                }
                else
                { ret.Add(new OpenCvSharp.Rect((int)ycmin + 135, y0, 46, y1)); }
                
            }

            return ret;
        }

        public static List<OpenCvSharp.Rect> GetCoordHighPT(Mat blurred, Mat edged)
        {
            var rectlist = new List<OpenCvSharp.Rect>();

            var edged23 = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged23, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 23, 20);
            var struc = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(3, 3));
            var erodemat = new Mat();
            Cv2.Erode(edged, erodemat, struc);

            var matlist = new List<Mat>();
            matlist.Add(edged);
            matlist.Add(edged23);
            matlist.Add(erodemat);
            foreach (var m in matlist)
            {
                var outmat = new Mat();
                var ids = OutputArray.Create(outmat);
                var cons = new Mat[] { };
                Cv2.FindContours(m, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                var idx1 = 0;
                foreach (var item in cons)
                {
                    idx1++;

                    var crect = Cv2.BoundingRect(item);

                    if (crect.Width > 36 && crect.Width <= 50 && crect.Height > 54 && crect.Height <= 65 && crect.Y > 8)
                    {
                        //Cv2.Rectangle(coorenhance, crect, new Scalar(0, 255, 0));
                        //using (new Window("xyenhance4", coorenhance))
                        //{
                        //    Cv2.WaitKey();
                        //}
                        rectlist.Add(crect);
                    }
                }
            }//end foreach

            return rectlist;
        }

        private static List<double> GetCoordWidthPT_W(Mat mat, Mat edged)
        {
            //var denoisemat = new Mat();
            //Cv2.FastNlMeansDenoisingColored(mat, denoisemat, 10, 10, 7, 21);

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(edged, null, out kazeKeyPoints, kazeDescriptors);

            var hl = 0.25 * mat.Height;
            var hh = 0.75 * mat.Height;
            var wl = 10;
            var wh = mat.Width - 10;
            var hptlist = new List<KeyPoint>();
            foreach (var pt in kazeKeyPoints)
            {
                if (pt.Pt.Y >= hl && pt.Pt.Y <= hh
                    && pt.Pt.X >= wl && pt.Pt.X <= wh)
                {
                    hptlist.Add(pt);
                }
            }

            //var wptlist = hptlist;

            var wptlist = new List<KeyPoint>();
            for (var idx = 15; idx < mat.Width;)
            {
                var ylist = new List<double>();

                var wlist = new List<KeyPoint>();
                foreach (var pt in hptlist)
                {
                    if (pt.Pt.X >= (idx - 15) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        ylist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 8 && (ylist.Max() - ylist.Min()) > 0.25 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 15;
            }

            var xlist = new List<double>();
            if (wptlist.Count() == 0)
            {
                return xlist;
            }

            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            var xlength = xlist.Max() - xlist.Min();
            var coordlength = 0.336 * xlength;
            var xmin = xlist.Min() + coordlength;
            var xmax = xlist.Max() - coordlength;

            //var xmid = (xlist.Max() + xlist.Min())/2;
            //var xmin = xmid - 0.166666 * xlength;
            //var xmax = xmid + 0.166666 * xlength;

            var xyptlist = new List<KeyPoint>();
            foreach (var pt in wptlist)
            {
                if (pt.Pt.X <= xmin || pt.Pt.X >= xmax)
                {
                    xyptlist.Add(pt);
                }
            }

            //var dstKaze = new Mat();
            //Cv2.DrawKeypoints(edged, xyptlist.ToArray(), dstKaze);

            //using (new Window("dstKaze", dstKaze))
            //{
            //    Cv2.WaitKey();
            //}

            xlist.Clear();
            foreach (var pt in xyptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            return xlist;
        }

        private static List<double> GetCoordWidthPT(Mat mat)
        {
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(mat, denoisemat, 10, 10, 7, 21);

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(denoisemat, null, out kazeKeyPoints, kazeDescriptors);
            
            var hl = 0.25* mat.Height;
            var hh = 0.75 * mat.Height;
            var wl = 10;
            var wh = mat.Width - 10;
            var hptlist = new List<KeyPoint>();
            foreach (var pt in kazeKeyPoints)
            {
                if (pt.Pt.Y >= hl && pt.Pt.Y <= hh 
                    && pt.Pt.X >= wl && pt.Pt.X <= wh)
                {
                    hptlist.Add(pt);
                }
            }

            //var wptlist = hptlist;

            var wptlist = new List<KeyPoint>();
            for (var idx = 15; idx < mat.Width;)
            {
                var ylist = new List<double>();

                var wlist = new List<KeyPoint>();
                foreach (var pt in hptlist)
                {
                    if (pt.Pt.X >= (idx - 15) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        ylist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 8 && (ylist.Max() - ylist.Min()) > 0.25 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 15;
            }

            var xlist = new List<double>();
            if (wptlist.Count() == 0)
            {
                return xlist;
            }

            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            var xlength = xlist.Max() - xlist.Min();
            var coordlength = 0.336 * xlength;
            var xmin = xlist.Min() + coordlength;
            var xmax = xlist.Max() - coordlength;

            //var xmid = (xlist.Max() + xlist.Min())/2;
            //var xmin = xmid - 0.166666 * xlength;
            //var xmax = xmid + 0.166666 * xlength;

            var xyptlist = new List<KeyPoint>();
            foreach (var pt in wptlist)
            {
                if (pt.Pt.X <= xmin || pt.Pt.X >= xmax)
                {
                    xyptlist.Add(pt);
                }
            }

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(denoisemat, xyptlist.ToArray(), dstKaze);

            using (new Window("dstKaze", dstKaze))
            {
                Cv2.WaitKey();
            }

            xlist.Clear();
            foreach (var pt in xyptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            return xlist;
        }

        private static List<OpenCvSharp.Rect> GetGuessXlistCircle2168(Mat edged,List<int> cwlist, List<int> y0list, List<int> y1list, List<int> wavglist)
        {
            var rectlist = new List<OpenCvSharp.Rect>();
            var xlist = GetCoordWidthPT(edged);
            var leftedge = xlist.Min();
            var rightedge = xlist.Max();

            var wavg = wavglist.Average() + 2;
            

            var assumexlist = new List<int>();
            for (var idx = 0; idx < 8; idx++)
            { assumexlist.Add(-1); }

            foreach (var val in cwlist)
            {
                if (val > (leftedge - 0.5 * wavg) && val < (leftedge + 0.5 * wavg))
                { assumexlist[0] = val; }
                if (val > (leftedge + 0.5 * wavg) && val < (leftedge + 1.5 * wavg))
                { assumexlist[1] = val; }
                if (val > (leftedge + 1.5 * wavg) && val < (leftedge + 2.5 * wavg))
                { assumexlist[2] = val; }
                if (val > (leftedge + 2.5 * wavg) && val < (leftedge + 3.5 * wavg))
                { assumexlist[3] = val; }

                if (val > (rightedge - 1.5 * wavg) && val < (rightedge - 0.5 * wavg))
                { assumexlist[7] = val; }
                if (val > (rightedge - 2.5 * wavg) && val < (rightedge - 1.5 * wavg))
                { assumexlist[6] = val; }
                if (val > (rightedge - 3.5 * wavg) && val < (rightedge - 2.5 * wavg))
                { assumexlist[5] = val; }
                if (val > (rightedge - 4.5 * wavg) && val < (rightedge - 3.5 * wavg))
                { assumexlist[4] = val; }
            }

            if (assumexlist[0] == -1)
            {
                if (assumexlist[1] != -1) { assumexlist[0] = assumexlist[1] - (int)wavg - 1; }
                else { assumexlist[0] = (int)leftedge - 2; }
            }
            if (assumexlist[1] == -1)
            {
                if (assumexlist[2] != -1) { assumexlist[1] = assumexlist[2] - (int)wavg - 1; }
                else { assumexlist[1] = assumexlist[0] + (int)wavg; }
            }
            if (assumexlist[2] == -1)
            {
                if (assumexlist[3] != -1)
                { assumexlist[2] = assumexlist[3] - (int)wavg - 1; }
                else
                { assumexlist[2] = assumexlist[1] + (int)wavg; }
            }
            if (assumexlist[3] == -1)
            { assumexlist[3] = assumexlist[2] + (int)wavg; }

            if (assumexlist[7] == -1)
            {
                if (assumexlist[6] != -1)
                { assumexlist[7] = assumexlist[6] + (int)wavg; }
                else
                { assumexlist[7] = (int)rightedge - (int)wavg - 1; }
            }
            if (assumexlist[6] == -1)
            {
                if (assumexlist[5] != -1)
                { assumexlist[6] = assumexlist[5] + (int)wavg; }
                else
                { assumexlist[6] = assumexlist[7] - (int)wavg - 1; }
            }
            if (assumexlist[5] == -1)
            {
                if (assumexlist[4] != -1)
                { assumexlist[5] = assumexlist[4] + (int)wavg; }
                else
                { assumexlist[5] = assumexlist[6] - (int)wavg - 1; }
            }
            if (assumexlist[4] == -1)
            { assumexlist[4] = assumexlist[5] - (int)wavg - 2; }

            var h0avg = (int)y0list.Average() - 1;
            var h1avg = (int)y1list.Average() + 1;

            rectlist.Clear();
            for (var idx = 0; idx < 8; idx++)
            {
                if (idx == 3)
                {
                    rectlist.Add(new OpenCvSharp.Rect(assumexlist[idx] - 1, h0avg, (int)wavg + 2, h1avg));
                }
                else if (idx == 7)
                { rectlist.Add(new OpenCvSharp.Rect(assumexlist[idx] - 1, h0avg, (int)wavg + 2, h1avg)); }
                else
                { rectlist.Add(new OpenCvSharp.Rect(assumexlist[idx] - 1, h0avg, assumexlist[idx + 1] - assumexlist[idx], h1avg)); }
            }

            return rectlist;
        }

        public static List<OpenCvSharp.Rect> GetPossibleXlistCircle2168(Mat edged,Mat xyenhance4, List<double> radlist)
        {
            var outmat = new Mat();
            var ids = OutputArray.Create(outmat);
            var cons = new Mat[] { };
            Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            var cwlist = new List<int>();
            var y0list = new List<int>();
            var y1list = new List<int>();
            var wavglist = new List<int>();
            var rectlist = new List<OpenCvSharp.Rect>();


            var idx1 = 0;
            foreach (var item in cons)
            {
                idx1++;

                var crect = Cv2.BoundingRect(item);
                if (crect.Width > 38 && crect.Width < 64 && crect.Height > 50 && crect.Height < 100)
                {
                    //radlist.Add(crect.Width);
                    //Cv2.Rectangle(xyenhance4, crect, new Scalar(0, 255, 0));
                    //using (new Window("xyenhance4", xyenhance4))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    rectlist.Add(crect);
                    y0list.Add(crect.Y);
                    wavglist.Add(crect.Width);
                    cwlist.Add(crect.X);
                    y0list.Add(crect.Y);
                    y1list.Add(crect.Height);
                }
            }//end foreach
            
            if (rectlist.Count == 8)
            {
                rectlist.Sort(delegate (OpenCvSharp.Rect r1, OpenCvSharp.Rect r2) {
                    return r1.X.CompareTo(r2.X);
                });
                return rectlist;
            }
            else
            {
                cwlist.Sort();
                return GetGuessXlistCircle2168(edged, cwlist, y0list, y1list, wavglist);
            }
        }


        private static List<List<double>> GetDetectPointUniq(Mat mat)
        {
            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(mat, null, out kazeKeyPoints, kazeDescriptors);
            var xlist = new List<double>();
            var ylist = new List<double>();
            foreach (var pt in kazeKeyPoints)
            {
                xlist.Add(pt.Pt.X);
                ylist.Add(pt.Pt.Y);
            }
            ret.Add(xlist);
            ret.Add(ylist);

            //var dstKaze = new Mat();
            //Cv2.DrawKeypoints(mat, kazeKeyPoints, dstKaze);

            //using (new Window("dstKaze", dstKaze))
            //{
            //    Cv2.WaitKey();
            //}

            return ret;
        }

        public void Rununiq(string imgpath)
        {

            Mat src = Cv2.ImRead(imgpath, ImreadModes.Color);
             var detectsize = GetDetectPointUniq(src);
            var srcrealimg = src.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            var xyenhance = new Mat();
            Cv2.DetailEnhance(srcrealimg, xyenhance);

            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance, denoisemat, 10, 10, 7, 21);


            var xyenhgray = new Mat();
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            //Cv2.MedianBlur(xyenhgray, xyenhgray, 5);

            //var blurred = new Mat();
            //Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            //var edged = new Mat();
            //Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);
            //using (new Window("edged", edged))
            //{
            //    Cv2.WaitKey();
            //}

            var high = srcrealimg.Height;
            var minrad = (int)(high * 0.12) - 10;
            var maxrad = (int)(high * 0.155) + 10;

            var hl = 0.375 * high;
            var hh = 0.625 * high;

            ////edged.Rows / 8, 100, 70, 10, 800
            var circles = Cv2.HoughCircles(xyenhgray, HoughMethods.Gradient, 1, xyenhgray.Rows / 4, 100, 70, minrad, maxrad);
            foreach (var c in circles)
            {
                var centerh = (int)c.Center.Y;
                if (centerh >= hl && centerh <= hh)
                {
                    Cv2.Circle(xyenhance, (int)c.Center.X, (int)c.Center.Y, (int)c.Radius, new Scalar(0, 255, 0), 3);
                    using (new Window("srcimg", xyenhance))
                    {
                        Cv2.WaitKey();
                    }
                }
            }

        }

        public string CheckRegion2168(Mat checkregion)
        {
            var xyenhance = new Mat();
            Cv2.DetailEnhance(checkregion, xyenhance);

            var denoisemat1 = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance, denoisemat1, 10, 10, 7, 21);
            xyenhance = denoisemat1;

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            using (new Window("check-edged", edged))
            {
                Cv2.WaitKey();
            }

            for (var idx = 1; idx < edged.Height - 3; idx=idx+3)
            {
                var snapmat = edged.SubMat(idx, idx+3, 0, edged.Width);
                var cnt = snapmat.CountNonZero();
                if (cnt > 200)
                {
                    return "2x1";
                }
            }

            return "2168";
        }

        public void Runcircle2168(string imgpath, List<double> ratelist, List<double> radlist,bool trad=true)
        {
            Mat srccolor = Cv2.ImRead(imgpath, ImreadModes.Color);

            //var angle = GetAngle2168(imgpath);
            //if (angle >= 0.7 && angle <= 359.3)
            //{
            //    var center = new Point2f(srccolor.Width / 2, srccolor.Height / 2);
            //    var m = Cv2.GetRotationMatrix2D(center, angle, 1);
            //    var outxymat = new Mat();
            //    Cv2.WarpAffine(srccolor, outxymat, m, new Size(srccolor.Width, srccolor.Height));
            //    srccolor = outxymat;
            //}

            var detectsize = GetDetectPoint(srccolor);
            var srcrealimg = srccolor.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            using (new Window("srcrealimg", srcrealimg))
            {
                Cv2.WaitKey();
            }

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);
            var srcblurred = new Mat();
            Cv2.GaussianBlur(srcgray, srcblurred, new Size(5, 5), 0);
            var srcedged = new Mat();
            Cv2.AdaptiveThreshold(srcblurred, srcedged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows / 4, 100, 80, 30, 70);

            var lowbond = srcrealimg.Height * 0.2;
            var upbond = srcrealimg.Height * 0.8;
            var midbond = srcrealimg.Height * 0.5;

            var filtercircles = new List<CircleSegment>();
            foreach (var c in circles)
            {
                if (c.Center.Y > lowbond && c.Center.Y < upbond)
                {
                    filtercircles.Add(c);
                }
            }

            if(filtercircles.Count > 0)
            {
                var CP = filtercircles[0];
                var hg = srcrealimg.Height;

                var lines = Cv2.HoughLinesP(srcedged, 1, Math.PI / 180.0, 50, 80, 5);
                var filterline = new List<LineSegmentPoint>();
                foreach (var line in lines)
                {
                    var degree = Math.Atan2((line.P2.Y - line.P1.Y), (line.P2.X - line.P1.X));
                    var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);
                    var xlen = Math.Abs(line.P2.X - line.P1.X);
                    var ylen = CP.Center.Y - line.P1.Y;
                    if (CP.Center.Y < midbond)
                    { ylen = line.P1.Y - CP.Center.Y; }

                    if ((d360 <= 4 || d360 >= 356))
                    {
                        if (xlen >= 140 && xlen < 240 
                        && (ylen >= 170 && ylen <= 210))
                        {
                            filterline.Add(line);
                        }
                    }
                }

                if (filterline.Count > 0)
                {
                    var checkregion = new Mat();
                    var ylist = new List<int>();
                    foreach (var p in filterline)
                    { ylist.Add(p.P1.Y);ylist.Add(p.P2.Y); }
                    var boundy = (int)ylist.Average();
                    var midx = (int)CP.Center.X;

                    //if (CP.Center.Y > midbond)
                    //{
                    //    var colstart = midx - 15;
                    //    var colend = midx + 15;
                    //    var rowstart = boundy + 15;
                    //    var rowend = boundy + 40;
                    //    checkregion = srcrealimg.SubMat(rowstart, rowend, colstart, colend);
                    //}
                    //else
                    //{
                    //    var colstart = midx - 15;
                    //    var colend = midx + 15;
                    //    var rowstart = boundy - 40;
                    //    var rowend = boundy - 15;

                    //    checkregion = srcrealimg.SubMat(rowstart, rowend, colstart, colend);
                    //}

                    //using (new Window("checkregion", checkregion))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    //var revsion = CheckRegion2168(checkregion);

                    var coormat = new Mat();
                    radlist.Add(CP.Radius);
                    ratelist.Add((double)Math.Abs(CP.Center.Y - boundy)/(double)CP.Radius);

                    if (CP.Center.Y > midbond)
                    {
                        var colstart = midx - 120;
                        var colend = midx + 120;
                        var rowstart = boundy + 3;
                        var rowend = boundy + 58;

                        if (colstart < 0 || colend > srcrealimg.Width)
                        {
                            if (srcrealimg.Width - midx > midx)
                            {
                                colstart = 1;
                                colend = 2*midx-1;
                            }
                            else
                            {
                                colstart = 2*midx - srcrealimg.Width + 1;
                                colend = srcrealimg.Width - 1;
                            }
                            rowstart = boundy + 3;
                            rowend = boundy + 46;
                        }
                        coormat = srcrealimg.SubMat(rowstart, rowend, colstart, colend);
                    }
                    else
                    {
                        var colstart = midx - 120;
                        var colend = midx + 120;
                        var rowstart = boundy - 58;
                        var rowend = boundy-3;

                        if (colstart < 0 || colend > srcrealimg.Width)
                        {
                            if (srcrealimg.Width - midx > midx)
                            {
                                colstart = 1;
                                colend = 2 * midx - 1;
                            }
                            else
                            {
                                colstart = 2 * midx - srcrealimg.Width + 1;
                                colend = srcrealimg.Width - 1;
                            }
                            rowstart = boundy - 46;
                            rowend = boundy - 3;
                        }
                        coormat = srcrealimg.SubMat(rowstart, rowend, colstart, colend);
                        var outxymat = new Mat();
                        Cv2.Transpose(coormat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        Cv2.Transpose(outxymat, outxymat);
                        Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                        coormat = outxymat;
                    }

                    using (new Window("coormat", coormat))
                    {
                        Cv2.WaitKey();
                    }

                    //var ylen = CP.Center.Y - boundy;
                    //if (CP.Center.Y < midbond)
                    //{ ylen = boundy - CP.Center.Y; }

                    var charmatlist = new List<Mat>();
                    if (trad)
                    { charmatlist = Get2168MatList(coormat); }
                    else
                    { charmatlist = Get2168MatList1(coormat); }

                    var idx = 0;
                    foreach (var cm in charmatlist)
                    {
                        if (idx == 0)
                        { idx++; continue; }

                        var tcm = new Mat();
                        cm.ConvertTo(tcm, MatType.CV_32FC1);
                        var tcmresize = new Mat();
                        Cv2.Resize(tcm, tcmresize, new Size(50, 50), 0, 0, InterpolationFlags.Linear);

                        using (new Window("cmresize1" + idx, tcmresize))
                        {
                            Cv2.WaitKey();
                        }

                        idx++;
                    }//end foreach
                }//end line
            }//end circle
        }

        private static List<Mat> Get2168MatList(Mat coordmat)
        {
            var cmatlist = new List<Mat>();

            var xyenhance = coordmat;


            {
                xyenhance = new Mat();
                Cv2.DetailEnhance(coordmat, xyenhance);

                var denoisemat1 = new Mat();
                Cv2.FastNlMeansDenoisingColored(xyenhance, denoisemat1, 10, 10, 7, 21);
                xyenhance = denoisemat1;
            }

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            using (new Window("xyenhance4x", xyenhance4x))
            {
                Cv2.WaitKey();
            }

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            using (new Window("denoisemat", denoisemat))
            {
                Cv2.WaitKey();
            }

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(7, 7), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            var rectlist = Get2168Rect(edged, xyenhance4x);
            //if (ylen < 190)
            //{ Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15); }

            cmatlist.Add(xyenhance);
            foreach (var rect in rectlist)
            {
                if (rect.X < 0 || rect.Y < 0
                || ((rect.X + rect.Width) > edged.Width)
                || ((rect.Y + rect.Height) > edged.Height))
                {
                    cmatlist.Clear();
                    return cmatlist;
                }

                cmatlist.Add(edged.SubMat(rect));
            }

            return cmatlist;
        }

        private static List<Mat> Get2168MatList1(Mat coordmat)
        {
            var cmatlist = new List<Mat>();
            var xyenhance = coordmat;
            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            using (new Window("xyenhance4x", xyenhance4x))
            {
                Cv2.WaitKey();
            }

            var lowspec = new Scalar(23, 0, 0);
            //var highspec = new Scalar(100, 67, 65);
            var highspec = new Scalar(143, 63, 56);

            var coordhsv = new Mat();
            Cv2.CvtColor(xyenhance4x, coordhsv, ColorConversionCodes.BGR2RGB);

            var edged = coordhsv.InRange(lowspec, highspec);

            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            var rectlist = Get2168Rect(edged, xyenhance4x);
            //if (ylen < 190)
            //{ Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15); }

            cmatlist.Add(xyenhance);
            foreach (var rect in rectlist)
            {
                if (rect.X < 0 || rect.Y < 0
                || ((rect.X + rect.Width) > edged.Width)
                || ((rect.Y + rect.Height) > edged.Height))
                {
                    cmatlist.Clear();
                    return cmatlist;
                }

                cmatlist.Add(edged.SubMat(rect));
            }

            return cmatlist;
        }

        public static List<Rect> Get2168Rect(Mat edged, Mat xyenhance4x)
        {
            var hl = GetHeighLow2168(edged);
            var hh = GetHeighHigh2168(edged);


            var dcl = hl;//(int)(hl + (hh - hl) * 0.1);
            var dch = hh;//(int)(hh - (hh - hl) * 0.1);
            var xxh = GetXXHigh2168(edged, dcl, dch);
            var yxl = GetYXLow2168(edged, dcl, dch);

            if (xxh == -1 || yxl == -1)
            {
                var xlist = GetCoordWidthPT2168(xyenhance4x, edged);
                var xmid = (xlist.Max() + xlist.Min()) / 2;
                var xcxlist = new List<double>();
                var ycxlist = new List<double>();
                foreach (var x in xlist)
                {
                    if (x < xmid)
                    { xcxlist.Add(x); }
                    else
                    { ycxlist.Add(x); }
                }
                xxh = (int)xcxlist.Max() + 2;
                yxl = (int)ycxlist.Min() - 2;
            }
            else
            {
                //avoid contamination at coord center
                //var wml = edged.Width / 2;
                //var xdist = wml - xxh;
                //var ydist = yxl - wml;
                //if (Math.Abs(xdist - ydist) > 90)
                //{
                //    if (xdist > ydist)
                //    { yxl = wml + xdist; }
                //    else
                //    { xxh = wml - ydist; }
                //}
            }

            var rectlist = new List<Rect>();

            var xxlist = GetXSplitList2168(edged, xxh, hl, hh);
            var flist = (List<int>)xxlist[0];
            var slist = (List<int>)xxlist[1];
            var y = hl - 5;
            var h = hh - hl + 7;

            if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                var left = slist[2] - fntw - 10;
                if (left < 0) { left = 1; }
                rectlist.Add(new Rect(left, y, fntw + 4, h));
                rectlist.Add(new Rect(slist[2] - 6, y, slist[1] - slist[2]+2, h));
                rectlist.Add(new Rect(slist[1] - 6, y, slist[0] - slist[1]+2, h));
                rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 8, h));
            }
            else if (slist.Count == 2)
            {
                var fntw = (int)flist.Average();
                var left = slist[1] - 2 * fntw - 10;
                if (left < 0) { left = 1; }
                rectlist.Add(new Rect(left, y, fntw + 4, h));
                rectlist.Add(new Rect(slist[1] - fntw - 6, y, fntw + 1, h));
                rectlist.Add(new Rect(slist[1] - 6, y, slist[0] - slist[1], h));
                rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 8, h));
            }
            else
            {
                if ((int)xxh - 226 > 0)
                { rectlist.Add(new Rect(xxh - 226, y, 56, h)); }
                else
                { rectlist.Add(new Rect(0, y, 56, h)); }
                rectlist.Add(new Rect(xxh - 174, y, 56, h));
                rectlist.Add(new Rect(xxh - 112, y, 56, h));
                rectlist.Add(new Rect(xxh - 54, y, 56, h));
            }

            var yxlist = GetYSplitList2168(edged, yxl, hl, hh);
            flist = (List<int>)yxlist[0];
            slist = (List<int>)yxlist[1];
            if (slist.Count == 4)
            {
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(slist[0] + 3, y, slist[1] - slist[0]+3, h));
                rectlist.Add(new Rect(slist[1] + 3, y, slist[2] - slist[1]+3, h));
                rectlist.Add(new Rect(slist[2] + 3, y, slist[3] - slist[2]+4, h));
            }
            else if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(slist[0] + 5, y, slist[1] - slist[0]+4, h));
                rectlist.Add(new Rect(slist[1] + 5, y, slist[2] - slist[1]+4, h));
                var left = slist[2] + 5;
                if (left + fntw + 4 > edged.Width)
                { left = edged.Width - fntw - 4; }
                rectlist.Add(new Rect(left, y, fntw + 4, h));
            }
            else if (slist.Count == 2)
            {
                var fntw = (int)flist.Average();
                rectlist.Add(new Rect(yxl - 3, y, slist[0] - yxl + 6, h));
                rectlist.Add(new Rect(slist[0] + 5, y, slist[1] - slist[0]+4, h));
                rectlist.Add(new Rect(slist[1] + 5, y, fntw + 4, h));
                var left = slist[1] + fntw + 12;
                if (left + fntw + 4 > edged.Width)
                { left = edged.Width - fntw - 4; }
                rectlist.Add(new Rect(left, y, fntw + 4, h));
            }
            else
            {
                rectlist.Add(new Rect(yxl - 2, y, 56, h));
                rectlist.Add(new Rect(yxl + 54, y, 56, h));
                rectlist.Add(new Rect(yxl + 113, y, 56, h));
                if ((yxl + 226) >= (edged.Cols - 1))
                { rectlist.Add(new Rect(yxl + 170, y, edged.Cols - yxl - 170, h)); }
                else
                { rectlist.Add(new Rect(yxl + 170, y, 54, h)); }
            }
            return rectlist;
        }

        private static List<double> GetCoordWidthPT2168(Mat mat, Mat edged)
        {

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(edged, null, out kazeKeyPoints, kazeDescriptors);

            var hl = 0.25 * mat.Height;
            var hh = 0.75 * mat.Height;
            var wl = 10;
            var wh = mat.Width - 10;
            var hptlist = new List<KeyPoint>();
            foreach (var pt in kazeKeyPoints)
            {
                if (pt.Pt.Y >= hl && pt.Pt.Y <= hh
                    && pt.Pt.X >= wl && pt.Pt.X <= wh)
                {
                    hptlist.Add(pt);
                }
            }

            //var wptlist = hptlist;

            var wptlist = new List<KeyPoint>();
            for (var idx = 15; idx < mat.Width;)
            {
                var ylist = new List<double>();

                var wlist = new List<KeyPoint>();
                foreach (var pt in hptlist)
                {
                    if (pt.Pt.X >= (idx - 15) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        ylist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 8 && (ylist.Max() - ylist.Min()) > 0.25 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 15;
            }

            var xlist = new List<double>();
            if (wptlist.Count() == 0)
            {
                return xlist;
            }

            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            var xlength = xlist.Max() - xlist.Min();
            var coordlength = 0.336 * xlength;
            var xmin = xlist.Min() + coordlength;
            var xmax = xlist.Max() - coordlength;

            var xyptlist = new List<KeyPoint>();
            foreach (var pt in wptlist)
            {
                if (pt.Pt.X <= xmin || pt.Pt.X >= xmax)
                {
                    xyptlist.Add(pt);
                }
            }

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(mat, xyptlist.ToArray(), dstKaze);

            using (new Window("dstKaze", dstKaze))
            {
                Cv2.WaitKey();
            }

            xlist.Clear();
            foreach (var pt in xyptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            return xlist;
        }

        public static int GetHeighLow2168(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.15);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.84);

            var xhl = 0;
            var yhl = 0;
            var ymidx = (int)(edged.Height * 0.4);
            for (var idx = ymidx; idx > 10; idx = idx - 2)
            {
                if (xhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhl = idx;
                    }
                }

                if (yhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhl = idx;
                    }
                }
            }

            var hl = xhl;
            if (yhl > hl)
            { hl = yhl; }

            return hl;
        }

        public static int GetHeighHigh2168(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.15);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.84);

            var xhh = 0;
            var yhh = 0;
            var ymidx = (int)(edged.Height * 0.4);
            for (var idx = ymidx; idx < edged.Height - 10; idx = idx + 2)
            {
                if (xhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhh = idx;
                    }
                }

                if (yhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhh = idx;
                    }
                }
            }

            var hh = 0;
            if (xhh > ymidx && yhh > ymidx)
            {
                if (yhh < xhh)
                { hh = yhh; }
                else
                { hh = xhh; }
            }
            else if (xhh > ymidx)
            { hh = xhh; }
            else if (yhh > ymidx)
            { hh = yhh; }
            else
            { hh = edged.Height - 10; }
            return hh;
        }

        public static int GetXXHigh2168(Mat edged, int dcl, int dch)
        {
            var ret = -1;
            var tm = 0;
            var wml = (int)(edged.Width * 0.25);
            var wmh = (int)(edged.Width * 0.5);            

            for (var idx = wmh; idx > wml; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    tm++;
                    if (ret == -1)
                    { ret = idx; }
                    else if(ret != -1 && tm > 8)
                    { return ret; }
                }
                else
                { ret = -1; tm = 0; }
            }

            return -1;
        }

        public static int GetYXLow2168(Mat edged, int dcl, int dch)
        {
            var ret = -1;
            var tm = 0;
            var wml = (int)(edged.Width * 0.5);
            var wmh = (int)(edged.Width * 0.75);

            for (var idx = wml; idx < wmh; idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    tm++;
                    if (ret == -1)
                    { ret = idx; }
                    else if(ret != -1 && tm > 8)
                    { return ret; }
                }
                else
                { ret = -1; tm = 0; }
            }
            return -1;
        }

        public static int GetXDirectSplit2168(Mat edged, int start, int end, int dcl, int dch,int previous)
        {
            var ret = -1;
            for (var idx = start; idx > end; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    if (ret == -1)
                    {
                        ret = idx;
                        if (previous-idx >= 48)
                        { return ret; }
                    }
                    else
                    { return ret; }
                }
                else
                { ret = -1; }
            }
            return -1;
        }

        public static int GetYDirectSplit2168(Mat edged, int start, int end, int dcl, int dch,int previous)
        {
            var ret = -1;
            for (var idx = start; idx < end; idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 4)
                {
                    if (ret == -1)
                    {
                        ret = idx;
                        if (idx - previous >= 48)
                        { return ret; }
                    }
                    else
                    { return ret; }
                }
                else
                { ret = -1; }
            }
            return -1;
        }

        public static List<object> GetXSplitList2168(Mat edged, int xxh, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spx1 = GetXDirectSplit2168(edged, xxh - 20, xxh - 20 - fntw, hl, hh,xxh);
            if (spx1 == -1) { return ret; }
            fntw = xxh - spx1 + 1;
            if (fntw >= 18 && fntw < 38)
            { spx1 = xxh - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx1);

            var spx2 = GetXDirectSplit2168(edged, spx1 - 28, spx1 - 28 - fntw, hl, hh,spx1);
            if (spx2 == -1) { return ret; }
            fntw = spx1 - spx2;
            if (fntw >= 18 && fntw < 38)
            { spx2 = spx1 - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx2);

            var spx3 = GetXDirectSplit2168(edged, spx2 - 28, spx2 - 28 - fntw, hl, hh,spx2);
            if (spx3 == -1) { return ret; }
            fntw = spx2 - spx3;
            if (fntw >= 18 && fntw < 38)
            { spx3 = spx2 - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx3);

            return ret;
        }
        public static List<object> GetYSplitList2168(Mat edged, int yxl, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spy1 = GetYDirectSplit2168(edged, yxl + 28, yxl + 28 + fntw, hl, hh,yxl);
            if (spy1 == -1) { return ret; }
            fntw = spy1 - yxl + 1;
            if (fntw >= 18 && fntw < 38)
            { spy1 = yxl + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy1);

            var spy2 = GetYDirectSplit2168(edged, spy1 + 28, spy1 + 28 + fntw, hl, hh,spy1);
            if (spy2 == -1) { return ret; }
            fntw = spy2 - spy1 + 1;
            if (fntw >= 18 && fntw < 38)
            { spy2 = spy1 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy2);

            var spy3 = GetYDirectSplit2168(edged, spy2 + 28, spy2 + 28 + fntw, hl, hh,spy2);
            if (spy3 == -1) { return ret; }
            fntw = spy3 - spy2 + 1;
            if (fntw >= 18 && fntw < 38)
            { spy3 = spy2 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy3);

            var spy4 = GetYDirectSplit2168(edged, spy3 + 28, edged.Width - 10, hl, hh,spy3);
            if (spy4 == -1) { return ret; }
            fntw = spy4 - spy3 + 1;
            if (fntw < 40)
            { return ret; }
            flist.Add(fntw); slist.Add(spy4);

            return ret;
        }


        private static double CheckCoordWidth(Mat mat)
        {
            var xlist = new List<double>();

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(mat, null, out kazeKeyPoints, kazeDescriptors);

            var hl = 0.25 * mat.Height;
            var hh = 0.75 * mat.Height;
            var wl = 5;
            var wh = mat.Width - 5;
            var hptlist = new List<KeyPoint>();
            foreach (var pt in kazeKeyPoints)
            {
                if (pt.Pt.Y >= hl && pt.Pt.Y <= hh
                    && pt.Pt.X >= wl && pt.Pt.X <= wh)
                {
                    hptlist.Add(pt);
                    xlist.Add(pt.Pt.X);
                }
            }

            //var dstKaze = new Mat();
            //Cv2.DrawKeypoints(mat, hptlist.ToArray(), dstKaze);

            //using (new Window("dstKaze", dstKaze))
            //{
            //    Cv2.WaitKey();
            //}

            return xlist.Max() - xlist.Min();
        }
        public void Run6(string imgpath,List<double> ratelist , List<double> radlist)
        {
            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            var srcgray = new Mat();
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var circles = Cv2.HoughCircles(srcgray, HoughMethods.Gradient, 1, srcgray.Rows/4, 100, 80, 40, 70);
            //foreach (var c in circles)
            //{
            //    Cv2.Circle(srcrealimg, (int)c.Center.X, (int)c.Center.Y, (int)c.Radius, new Scalar(0, 255, 0), 3);
            //    using (new Window("srcimg", srcrealimg))
            //    {
            //        Cv2.WaitKey();
            //    }
            //}

            var circlesegment = circles[0];
            var imgmidy = srcgray.Height / 2;
            var rad = circlesegment.Radius;
            var x0 = circlesegment.Center.X - 1.85 * rad;
            var x1 = circlesegment.Center.X + 1.85 * rad;
            var y0 = circlesegment.Center.Y - 3.56 * rad;
            if (y0 < 0) { y0 = 0; }
            var y1 = circles[0].Center.Y - 2.7 * rad;

            if (circlesegment.Center.Y < imgmidy)
            {
                x0 = circlesegment.Center.X - 1.85 * rad;
                x1 = circlesegment.Center.X + 1.85 * rad;
                y0 = circles[0].Center.Y + 2.7 * rad;
                y1 = circlesegment.Center.Y + 3.56 * rad;
                if (y1 > srcgray.Height) { y1 = srcgray.Height; }
            }


            var coordinatemat = srcrealimg.SubMat((int)y0, (int)y1, (int)x0, (int)x1);
            //using (new Window("submat", coordinatemat))
            //{
            //    Cv2.WaitKey();
            //}


            if (circlesegment.Center.Y < imgmidy)
            {
                var center = new Point2f(coordinatemat.Width / 2, coordinatemat.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(coordinatemat, outxymat, m, new Size(coordinatemat.Width, coordinatemat.Height));
                coordinatemat = outxymat;
            }



            var coordgray = new Mat();
            Cv2.CvtColor(coordinatemat, coordgray, ColorConversionCodes.BGR2GRAY);
            var coordblurred = new Mat();
            Cv2.GaussianBlur(coordgray, coordblurred, new Size(5, 5), 0);
            var coordedged = new Mat();
            Cv2.AdaptiveThreshold(coordblurred, coordedged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 7, 2);

            var coordwidth = CheckCoordWidth(coordedged);
            ratelist.Add(coordwidth);
            radlist.Add(circlesegment.Radius);

            //Cv2.Resize(coordedged, coordedged, new Size(coordedged.Width * 4, coordedged.Height * 4));

            //using (new Window("edged2", coordedged))
            //{
            //    Cv2.WaitKey();
            //}



            //var lines = Cv2.HoughLinesP(edged, 1, Math.PI / 180.0, 50, 40, 5);
            //foreach (var line in lines)
            //{
            //    var degree = Math.Atan2((line.P2.Y - line.P1.Y), (line.P2.X - line.P1.X));
            //    var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);

            //    if (d360 > 20 && d360 < 340)
            //    { continue; }

            //    Cv2.Line(srcimg, line.P1, line.P2, new Scalar(0, 255, 0), 3);
            //    using (new Window("srcimg1", srcimg))
            //    {
            //        Cv2.WaitKey();
            //    }

            //    if (d360 <= 1 || d360 >= 359)
            //    { break; }

            //    var center = new Point2f(srcimg.Width / 2, srcimg.Height / 2);
            //    var m = Cv2.GetRotationMatrix2D(center, d360, 1);
            //    var outxymat = new Mat();
            //    Cv2.WarpAffine(srcimg, outxymat, m, new Size(srcimg.Width, srcimg.Height));


            //    //Cv2.Line(srcimg,lines[0].P1,lines[0].P2,  new Scalar(0, 255, 0),3);
            //    using (new Window("edged1", outxymat))
            //    {
            //        Cv2.WaitKey();
            //    }

            //    src = new Mat();
            //    Cv2.CvtColor(outxymat, src, ColorConversionCodes.BGR2GRAY);
            //    blurred = new Mat();
            //    Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);
            //    edged = new Mat();
            //    Cv2.Canny(blurred, edged, 50, 200, 3, true);

            //    var circles = Cv2.HoughCircles(edged, HoughMethods.Gradient, 1, 40, 50, 50, 10, 300);
            //    foreach (var c in circles)
            //    {
            //        Cv2.Circle(outxymat, (int)c.Center.X, (int)c.Center.Y, (int)c.Radius, new Scalar(0, 255, 0), 3);
            //        using (new Window("srcimg", outxymat))
            //        {
            //            Cv2.WaitKey();
            //        }
            //    }

            //    var x0 = circles[0].Center.X - 1.8 * circles[0].Radius;
            //    var x1 = circles[0].Center.X + 1.8 * circles[0].Radius;
            //    var y0 = circles[0].Center.Y - 3.5* circles[0].Radius;
            //    if (y0 < 0) { y0 = 0; }
            //    var y1 = circles[0].Center.Y - 2.9 * circles[0].Radius;

            //    var submat = outxymat.SubMat((int)y0, (int)y1, (int)x0, (int)x1);
            //    using (new Window("submat", submat))
            //    {
            //        Cv2.WaitKey();
            //    }

            //    src = new Mat();
            //    Cv2.CvtColor(submat, src, ColorConversionCodes.BGR2GRAY);
            //    blurred = new Mat();
            //    Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);
            //    edged = new Mat();
            //    Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 11, 2);

            //    using (new Window("edged2", edged))
            //    {
            //        Cv2.WaitKey();
            //    }

            //    break;
            //}
        }

        public void Run5(string imgpath)
        {
            Mat srcimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var src = new Mat();
            Cv2.CvtColor(srcimg, src, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);
            var edged = new Mat();
            Cv2.Canny(blurred, edged, 50, 200, 3, true);

            var lines = Cv2.HoughLinesP(edged, 1, Math.PI / 180.0, 50,80, 5);
            foreach(var line in lines)
            {
                var degree = Math.Atan2((line.P2.Y - line.P1.Y), (line.P2.X - line.P1.X));
                var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);

                if (d360 > 20 && d360 < 340)
                { continue; }

                Cv2.Line(srcimg, line.P1, line.P2, new Scalar(0, 255, 0), 3);
                using (new Window("srcimg", srcimg))
                {
                    Cv2.WaitKey();
                }

                if (d360 <= 1 || d360 >= 359)
                { break; }

                var center = new Point2f(srcimg.Width / 2, srcimg.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, d360, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(srcimg, outxymat, m, new Size(srcimg.Width, srcimg.Height));


                //Cv2.Line(srcimg,lines[0].P1,lines[0].P2,  new Scalar(0, 255, 0),3);
                using (new Window("edged", outxymat))
                {
                    Cv2.WaitKey();
                }

                break;
            }
        }

        public void Run3()
        {
            var imgpath = @"E:\video\FAIL\NEW\158.png";
            //var imgpath = @"E:\video\FAIL\x\DIE-7.BMP";
            //var imgpath = @"E:\video\DIE-23.BMP";
            FindXYFlat1X2(imgpath);
        }

        private double GetAngle2168(string imgpath)
        {
            Mat srcimg = Cv2.ImRead(imgpath, ImreadModes.Color);

            var detectsize = GetDetectPoint(srcimg);
            srcimg = srcimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            var src = new Mat();
            Cv2.CvtColor(srcimg, src, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);
            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            //using (new Window("edged", edged))
            //{
            //    Cv2.WaitKey();
            //}
            var hg = srcimg.Height;
            var lines = Cv2.HoughLinesP(edged, 1, Math.PI / 180.0, 50, 80, 5);
            foreach (var line in lines)
            {
                var degree = Math.Atan2((line.P2.Y - line.P1.Y), (line.P2.X - line.P1.X));
                var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);
                if (d360 > 20 && d360 < 340)
                { continue; }

                if (d360 <= 4 || d360 >= 356)
                {
                    var xlen = line.P2.X - line.P1.X;
                    if (xlen > 180 && xlen < 240
                        && ((line.P1.Y > 30 && line.P1.Y < 100) || (line.P1.Y < hg - 30 && line.P1.Y > hg - 100)))
                    {
                        Cv2.Line(srcimg, line.P1, line.P2, new Scalar(0, 255, 0), 3);
                        using (new Window("srcimg", srcimg))
                        {
                            Cv2.WaitKey();
                        }
                        return d360;
                    }
                }
            }

            return 0;
        }

        private double GetAngle(string imgpath)
        {
            Mat srcimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var src = new Mat();
            Cv2.CvtColor(srcimg, src, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);
            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            //using (new Window("edged", edged))
            //{
            //    Cv2.WaitKey();
            //}

            var lines = Cv2.HoughLinesP(edged, 1, Math.PI / 180.0, 50, 80, 5);
            foreach (var line in lines)
            {
                var degree = Math.Atan2((line.P2.Y - line.P1.Y), (line.P2.X - line.P1.X));
                var d360 = (degree > 0 ? degree : (2 * Math.PI + degree)) * 360 / (2 * Math.PI);
                if (d360 > 20 && d360 < 340)
                { continue; }

                if (d360 <= 4 || d360 >= 356)
                {
                    return d360;
                    //var xlen = line.P2.X - line.P1.X;
                    //if (xlen > 180 && xlen < 240 
                    //    && line.P1.Y > 30 && line.P1.Y < 100
                    //    && line.P2.Y > 30 && line.P2.Y < 100)
                    //{
                    //    Cv2.Line(srcimg, line.P1, line.P2, new Scalar(0, 255, 0), 3);
                    //    using (new Window("srcimg", srcimg))
                    //    {
                    //        Cv2.WaitKey();
                    //    }
                    //}
                }
            }

            return 0;
       }

        public void Run5x1(string imgpath, OpenCvSharp.ML.ANN_MLP kmode)
        {
            //OGP-rect5x1,OGP-rect2x1,OGP-small5x1,OGP-circle2168
           

            var angle = GetAngle(imgpath);

            var xyrectlist = FindXYRect5X1(imgpath,angle, 25, 43, 4800, 8000);
            if (xyrectlist.Count > 0)
            {
                var charmatlist = CutCharRect5X1xxx(imgpath,angle, xyrectlist[0], 27, 50, 20, 50);
                var idx = 0;
                foreach (var cm in charmatlist)
                {
                    if (idx == 0)
                    { idx++; continue; }

                    var tcm = new Mat();
                    cm.ConvertTo(tcm, MatType.CV_32FC1);
                    var tcmresize = new Mat();
                    Cv2.Resize(tcm, tcmresize, new Size(30, 30), 0, 0, InterpolationFlags.Linear);

                    using (new Window("cmresize1" + idx, tcmresize))
                    {
                        Cv2.WaitKey();
                    }

                    if (kmode != null)
                    {
                        var resultmat = new Mat();
                        var stcm = tcmresize.Reshape(0, 1);

                        var matched = new Mat();
                        var m = kmode.Predict(stcm, matched);//,OpenCvSharp.ML.StatModel.Flags.RawOutput);
                        //kmode.FindNearest(stcm, 7, resultmat, matched);
                        var matchstr = matched.Dump();
                        var ms = matchstr.Split(new string[] { "[", "]", "," }, StringSplitOptions.RemoveEmptyEntries);
                        var mstr = "";
                        foreach (var s in ms)
                        {
                            mstr += UT.O2S((char)UT.O2I(s));
                        }

                        var blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
                        Cv2.PutText(blank, m.ToString(), new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                        using (new Window("blank", blank))
                        {
                            Cv2.WaitKey();
                        }
                    }


                    idx++;
                }
            }
        }

        public void Run6x1(string imgpath, OpenCvSharp.ML.KNearest kmode)
        {
            //OGP-rect5x1,OGP-rect2x1,OGP-small5x1,OGP-circle2168


            var angle = GetAngle(imgpath);

            var xyrectlist = FindXYRect5X1(imgpath, angle, 25, 43, 4800, 8000);
            if (xyrectlist.Count > 0)
            {
                var charmatlist = CutCharRect6X1xxx(imgpath, angle, xyrectlist[0], 27, 50, 20, 50);
                var idx = 0;
                foreach (var cm in charmatlist)
                {
                    if (idx == 0)
                    { idx++; continue; }

                    var tcm = new Mat();
                    cm.ConvertTo(tcm, MatType.CV_32FC1);
                    var tcmresize = new Mat();
                    Cv2.Resize(tcm, tcmresize, new Size(50, 50), 0, 0, InterpolationFlags.Linear);

                    using (new Window("cmresize1" + idx, tcmresize))
                    {
                        Cv2.WaitKey();
                    }

                    if (kmode != null)
                    {
                        var resultmat = new Mat();
                        var stcm = tcmresize.Reshape(1, 1);
                        var matched = new Mat();
                        kmode.FindNearest(stcm, 7, resultmat, matched);
                        var matchstr = matched.Dump();
                        var ms = matchstr.Split(new string[] { "[", "]", "," }, StringSplitOptions.RemoveEmptyEntries);
                        var mstr = "";
                        foreach (var s in ms)
                        {
                            mstr += UT.O2S((char)UT.O2I(s));
                        }

                        var blank = new Mat(new Size(240, 60), MatType.CV_32FC3, new Scalar(255, 255, 255));
                        Cv2.PutText(blank, mstr, new Point(6, 40), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                        using (new Window("blank", blank))
                        {
                            Cv2.WaitKey();
                        }
                    }


                    idx++;
                }
            }
        }

        public void Run2x1(string imgpath)
        {
            
            //var imgpath = @"E:\video\Die-18.BMP";
            //var xyrectlist = FindXYRect5X1(imgpath, 27, 43, 4800, 8000);

            //var imgpath = @"E:\video\FAIL\DIE-109.BMP";
            //xyrectlist = FindXYRect5X1(imgpath,27,43, 4800,8000);

            var xyrectlist = FindXYRect2X1(imgpath, 60, 100, 2.0, 3.0);
            if (xyrectlist.Count > 0)
            {
                var charmatlist = CutCharRect2X1(imgpath, xyrectlist[0]);


                //var samplex = new Mat();
                //var samples = new Mat();
                //samplex.ConvertTo(samples, MatType.CV_32FC1);

                //var responsarray = new List<int>();

                //var keylist = new List<char>();
                //keylist.Add('X');
                //keylist.Add('2');
                //keylist.Add('8');
                //keylist.Add('5');
                //keylist.Add('Y');
                //keylist.Add('2');
                //keylist.Add('6');
                //keylist.Add('0');

                var idx = 0;
                foreach (var cm in charmatlist)
                {
                    if (idx == 0)
                    { idx++; continue; }

                    var tcm = new Mat();
                    cm.ConvertTo(tcm, MatType.CV_32FC1);
                    var tcmresize = new Mat();
                    Cv2.Resize(tcm, tcmresize, new Size(50, 50), 0, 0, InterpolationFlags.Linear);
                    using (new Window("cmresize1" + idx, tcmresize))
                    {
                        Cv2.WaitKey();
                    }
                    //samples.PushBack(tcmresize.Reshape(1, 1));
                    //responsarray.Add((int)(keylist[idx]));
                    idx++;
                }

                //int[] rparray = responsarray.ToArray();
                //var responx = new Mat(rparray.Length, 1, MatType.CV_32SC1, rparray);
                //var respons = new Mat();
                //responx.ConvertTo(respons, MatType.CV_32FC1);

                //var kmode = OpenCvSharp.ML.KNearest.Create();
                //kmode.Train(samples, OpenCvSharp.ML.SampleTypes.RowSample,respons);
                //var trained = kmode.IsTrained();

                //var imgpath1 = @"E:\video\Die-18.BMP";
                //var xyrectlist1 = FindXYRect(imgpath1, 27, 43, 4800, 8000);
                //var charmatlist1 = CutCharRect(imgpath1, xyrectlist1[0], 30, 50, 20, 50);

                //var tcm1 = new Mat();
                //charmatlist1[3].ConvertTo(tcm1, MatType.CV_32FC1);
                //var tcmresize1 = new Mat();
                //Cv2.Resize(tcm1, tcmresize1, new Size(50, 50), 0, 0, InterpolationFlags.Linear);

                ////using (new Window("cmresize1", tcmresize1))
                ////{
                ////    Cv2.WaitKey();
                ////}

                //var stcm1 = tcmresize1.Reshape(1, 1);

                //var resultmat = new Mat();
                //var val = kmode.FindNearest(stcm1, 1, resultmat);

                //var rval = Convert.ToString((char)val);

                //var ival = (int)Convert.ToChar("2");

            }
        }


        private static List<Mat> GetCornerImg(Mat src)
        {
            var avsrcxylist = GetDetectPoint(src);
            var cstart = (int)avsrcxylist[0].Min();
            var cend = (int)avsrcxylist[0].Max();
            var rstart = (int)avsrcxylist[1].Min();
            var rend = (int)avsrcxylist[1].Max();

            var h4 = (rend - rstart) / 4 + 5;
            var w4 = (cend - cstart) / 4;

            Mat avsrc1 = src.SubMat(rstart, rstart + h4, cstart, cstart + w4);
            avsrc1 = avsrc1.SubMat((int)(avsrc1.Height * 0.35), avsrc1.Height, (int)(avsrc1.Width * 0.35), avsrc1.Width);

            Mat avsrc2 = src.SubMat(rend - h4, rend, cend - w4, cend);
            avsrc2 = avsrc2.SubMat(0, (int)(avsrc2.Height * 0.65), 0, (int)(avsrc2.Width * 0.65));

            var ret = new List<Mat>();
            ret.Add(avsrc1);
            ret.Add(avsrc2);
            return ret;
        }

        private static Mat GetReverseMask(Mat xyenhance)
        {
            var hsvsrc = new Mat();
            Cv2.CvtColor(xyenhance, hsvsrc, ColorConversionCodes.BGR2HSV);

            var brscal = new Scalar(153, 103, 104);
            var rgbbrmat = new Mat(1, 1, MatType.CV_8UC3, brscal);
            var hsvbrmat = new Mat();
            Cv2.CvtColor(rgbbrmat, hsvbrmat, ColorConversionCodes.RGB2HSV);

            var dkscal = new Scalar(138, 93, 93);
            var rgbdkmat = new Mat(1, 1, MatType.CV_8UC3, dkscal);
            var hsvdkmat = new Mat();
            Cv2.CvtColor(rgbdkmat, hsvdkmat, ColorConversionCodes.RGB2HSV);

            var mask = hsvsrc.InRange(hsvdkmat, hsvbrmat);

            var kernal = Mat.Ones(new Size(20, 20), MatType.CV_32F);
            Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernal);

            var reversemask = new Mat();
            Cv2.BitwiseNot(mask, reversemask);
            return reversemask;
        }

        private static List<OpenCvSharp.Rect> Try2GetImgRect(Mat cornerimg)
        {
            var ret = new List<OpenCvSharp.Rect>();

            var xyenhance = new Mat();
            Cv2.DetailEnhance(cornerimg, xyenhance);

            Cv2.Resize(xyenhance, xyenhance, new Size(300, 200));

            var reversemask = GetReverseMask(xyenhance);

            var fontimg = new Mat();
            Cv2.BitwiseAnd(xyenhance, xyenhance, fontimg, reversemask);

            var xyenhgray = new Mat();
            Cv2.CvtColor(fontimg, xyenhgray, ColorConversionCodes.BGR2GRAY);

            Cv2.Resize(xyenhgray, xyenhgray, new Size(xyenhgray.Width * 2, xyenhgray.Height * 2));

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 11, 2);

            var outmatx = new Mat();
            var ids = OutputArray.Create(outmatx);
            var cons = new Mat[] { };
            Cv2.FindContours(edged, out cons, ids, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var idx = 0;

            foreach (var c in cons)
            {
                idx++;
                var crect = Cv2.BoundingRect(c);

                if (crect.Height > 100 && crect.Height < 200 
                    && crect.Width > 200 && crect.Width < 400 
                    && ((crect.Y + crect.Height) < edged.Height - 30))
                {
                    ret.Add(crect);
                }
            }

            return ret;
        }


        private void FindXYFlat1X2(string file)
        {
            Mat src = Cv2.ImRead(file, ImreadModes.Color);
            var cornermats = GetCornerImg(src);
            var lefttopmat = cornermats[0];
            var rightbtmat = cornermats[1];


            var rectlist = Try2GetImgRect(lefttopmat);
            if (rectlist.Count == 1)
            {
                var xyenhance = new Mat();
                Cv2.DetailEnhance(lefttopmat, xyenhance);
                Cv2.Resize(xyenhance, xyenhance, new Size(600, 400));
                var fontmat = xyenhance.SubMat(rectlist[0]);
                var XFont = fontmat.SubMat(0, fontmat.Rows, 0, 90);
                using (new Window("XFont", XFont))
                {
                    Cv2.WaitKey();
                }
            }

            rectlist = Try2GetImgRect(rightbtmat);
            if (rectlist.Count == 1)
            {
                var xyenhance = new Mat();
                Cv2.DetailEnhance(rightbtmat, xyenhance);
                Cv2.Resize(xyenhance, xyenhance, new Size(600, 400));
                var fontmat = xyenhance.SubMat(rectlist[0]);

                var center = new Point2f(fontmat.Width / 2, fontmat.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(fontmat, outxymat, m, new Size(fontmat.Width, fontmat.Height));
                fontmat = outxymat;

                var XFont = fontmat.SubMat(0, fontmat.Rows, 0, 90);
                using (new Window("XFont", XFont))
                {
                    Cv2.WaitKey();
                }
            }
        }

        private List<OpenCvSharp.Rect> FindXYRect2X1(string file, int heighlow, int heighhigh, double ratelow, double ratehigh)
        {
            Mat srcorgimg = Cv2.ImRead(file, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            //var srcenhance = new Mat();
            //Cv2.DetailEnhance(srcrealimg, srcenhance);

            var srcgray = new Mat();
            //var denoisemat = new Mat();
            //Cv2.FastNlMeansDenoisingColored(srcenhance, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(srcrealimg, srcgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(srcgray, blurred, new Size(5, 5), 0);

            var ret = new List<OpenCvSharp.Rect>();
            var cannyflags = new List<bool>();
            cannyflags.Add(true);
            cannyflags.Add(false);
            foreach (var cflag in cannyflags)
            {
                var edged = new Mat();
                Cv2.Canny(blurred, edged, 50, 200, 3, cflag);

                using (new Window("edged", edged))
                {
                    Cv2.WaitKey();
                }



                var outmat = new Mat();
                var ids = OutputArray.Create(outmat);
                var cons = new Mat[] { };
                Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
                var conslist = cons.ToList();
                //conslist.Sort(delegate (Mat obj1, Mat obj2)
                //{
                //    return Cv2.ContourArea(obj2).CompareTo(Cv2.ContourArea(obj1));
                //});

                var idx = 0;
                foreach (var item in conslist)
                {
                    idx++;

                    var rect = Cv2.BoundingRect(item);
                    var whrate = (float)rect.Width / (float)rect.Height;
                    var a = rect.Width * rect.Height;

                    if (whrate > ratelow && whrate < ratehigh
                        && rect.Height > heighlow && rect.Height < heighhigh)
                    {
                        var xymat = srcrealimg.SubMat(rect);
                        using (new Window("xymat" + idx, xymat))
                        {
                            Cv2.WaitKey();
                        }

                        if (ret.Count > 0)
                        {
                            if (a > ret[0].Width * ret[0].Height)
                            {
                                ret.Clear();
                                ret.Add(rect);
                            }
                        }
                        else
                        { ret.Add(rect); }
                    }
                }

            }

            if (ret.Count > 0)
            {
                var coord = srcrealimg.SubMat(ret[0]);
                var midx = (int)(coord.Width / 2);
                var midy = (int)(coord.Height / 2);
                var checkregion = coord.SubMat(midy - 15, midy + 15, midx - 20, midx + 20);
                using (new Window("check-edged", checkregion))
                {
                    Cv2.WaitKey();
                }
                var ck = CheckRegion2x1(checkregion);
                if (ck)
                { return ret; }
                else
                { return new List<Rect>(); }
            }
            return ret;
        }


        public bool CheckRegion2x1(Mat checkregion)
        {
            var xyenhance = new Mat();
            Cv2.DetailEnhance(checkregion, xyenhance);

            var denoisemat1 = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance, denoisemat1, 10, 10, 7, 21);
            xyenhance = denoisemat1;

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));
            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            using (new Window("check-edged", edged))
            {
                Cv2.WaitKey();
            }

            for (var idx = 0; idx < edged.Width - 3; idx = idx + 3)
            {
                var snapmat = edged.SubMat(0, edged.Height, idx, idx+3);
                var cnt = snapmat.CountNonZero();
                if (cnt > 65)
                {
                    return true;
                }
            }

            return false;
        }

        private List<Mat> CutCharRect2X1(string imgpath, OpenCvSharp.Rect xyrect)
        {
            var cmatlist = new List<Mat>();

            Mat srcorgimg = Cv2.ImRead(imgpath, ImreadModes.Color);
            var detectsize = GetDetectPoint(srcorgimg);
            var srcrealimg = srcorgimg.SubMat((int)detectsize[1].Min(), (int)detectsize[1].Max(), (int)detectsize[0].Min(), (int)detectsize[0].Max());

            //var srcenhance = new Mat();
            //Cv2.DetailEnhance(srcrealimg, srcenhance);

            //var denoisemat = new Mat();
            //Cv2.FastNlMeansDenoisingColored(srcenhance, denoisemat, 10, 10, 7, 21);

            var xymat = srcrealimg.SubMat(xyrect);

            using (new Window("xymat", xymat))
            {
                Cv2.WaitKey();
            }

            var srcmidy = (detectsize[1].Max() + detectsize[1].Min()) / 2;

            if ((xyrect.Y+(xyrect.Height/2)) < srcmidy)
            {
                var outxymat = new Mat();
                Cv2.Transpose(xymat, outxymat);
                Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                Cv2.Transpose(outxymat, outxymat);
                Cv2.Flip(outxymat, outxymat, FlipMode.Y);
                //var center = new Point2f(xymat.Width / 2, xymat.Height / 2);
                //var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                //var outxymat = new Mat();
                //Cv2.WarpAffine(xymat, outxymat, m, new Size(xymat.Width, xymat.Height));
                xymat = outxymat;
            }


            var newxymat = xymat.SubMat(4, xymat.Rows - 20, (int)(xymat.Cols * 0.63), xymat.Cols - 12);

            using (new Window("newxymat", newxymat))
            {
                Cv2.WaitKey();
            }

            var xyenhance = new Mat();
            Cv2.DetailEnhance(newxymat, xyenhance);

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));

            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat2 = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat2, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat2, xyenhgray, ColorConversionCodes.BGR2GRAY);

            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);

            var xypts = GetCoordPT2X1(edged);
            var xmin = xypts[0].Min();

            if (xmin < 6)
            {
                newxymat = xymat.SubMat(3, xymat.Rows - 20, (int)(xymat.Cols * 0.63-2.5), xymat.Cols - 12);

                xyenhance = new Mat();
                Cv2.DetailEnhance(newxymat, xyenhance);
                xyenhance4x = new Mat();
                Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));
                Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

                xyenhgray = new Mat();
                denoisemat2 = new Mat();
                Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat2, 10, 10, 7, 21);
                Cv2.CvtColor(denoisemat2, xyenhgray, ColorConversionCodes.BGR2GRAY);

                blurred = new Mat();
                Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);
                edged = new Mat();
                Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);
            }


            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }


            var rectlist = GetCharRect2X1(xyenhance4x, edged);
            cmatlist.Add(xyenhance);
            foreach (var r in rectlist)
            { cmatlist.Add(edged.SubMat(r)); }

            return cmatlist;
        }

        private List<OpenCvSharp.Rect> GetCharRect2X1(Mat xyenhance4x,Mat edged)
        {
            var xypts = GetCoordPT2X1(edged);
            var cbox = GetCoorBond2x1(xyenhance4x, edged, 40, 56);

            var y0list = new List<int>();
            var y1list = new List<int>();
            var x0list = new List<int>();
            foreach (var bx in cbox)
            {
                if (bx.Y < 65)
                { y0list.Add(bx.Y); }
                else
                { y1list.Add(bx.Y); }
                x0list.Add(bx.X);
            }

            var y0 = (int)xypts[1].Min() - 2;
            if (y0list.Count > 0)
            { y0 = y0list.Min() - 2; }
            var h0 = 82;
            if (y0 < 0) { y0 = 0; }

            var y1 = y0 + 100;
            if (y0list.Count == 0 && y1list.Count > 0)
            { y1 = y1list.Min() - 2; }
            var h1 = 82;
            if ((y1 + h1) > edged.Height)
            { h1 = edged.Height - y1 - 1; }

            var x0 = (int)xypts[0].Min() - 2;
            if (x0list.Count > 0)
            {
                var xmin = x0list.Min();
                if (xmin > (x0 - 15) && xmin < (x0 + 15))
                { x0 = xmin - 2; }
            }
            if (x0 < 0)
            { x0 = 0; }

            var x1 = x0 + 54;
            var x2 = x0 + 106;
            var x3 = x0 + 160;
            var wd = 54;

            var rectlist = new List<OpenCvSharp.Rect>();
            rectlist.Add(new OpenCvSharp.Rect(x0, y0, wd + 1, h0));
            rectlist.Add(new OpenCvSharp.Rect(x1, y0, wd, h0));
            rectlist.Add(new OpenCvSharp.Rect(x2, y0, wd, h0));
            if (x3 + wd > edged.Width)
            {
                var wd1 = edged.Width - x3 - 1;
                rectlist.Add(new OpenCvSharp.Rect(x3, y0, wd1, h0));
            }
            else
            { rectlist.Add(new OpenCvSharp.Rect(x3, y0, wd, h0)); }

            rectlist.Add(new OpenCvSharp.Rect(x0, y1, wd + 1, h1));
            rectlist.Add(new OpenCvSharp.Rect(x1, y1, wd, h1));
            rectlist.Add(new OpenCvSharp.Rect(x2, y1, wd, h1));
            if (x3 + wd > edged.Width)
            {
                var wd1 = edged.Width - x3 - 1;
                rectlist.Add(new OpenCvSharp.Rect(x3, y1, wd1, h1));
            }
            else
            { rectlist.Add(new OpenCvSharp.Rect(x3, y1, wd, h1)); }

            //foreach (var bx in cbox)
            //{
            //    var cnt = rectlist.Count;
            //    for (var ridx = 0; ridx < cnt; ridx++)
            //    {
            //        var r = rectlist[ridx];
            //        if (bx.X > (r.X - 16) && bx.X < (r.X + 16)
            //            && bx.Y > (r.Y - 16) && bx.Y < (r.Y + 16))
            //        {
            //            var x = ((bx.X - 2) > 0) ? (bx.X - 2) : 0;
            //            var y = ((bx.Y - 2) > 0) ? (bx.Y - 2) : 0;
            //            var w = bx.Width + 4;
            //            var h = bx.Height + 4;

            //            if ((x + w) > edged.Width)
            //            { w = edged.Width - x - 1; }
            //            if ((y + h) > edged.Height)
            //            { h = edged.Height - y - 1; }

            //            var crect = new OpenCvSharp.Rect(x, y, w, h);
            //            rectlist[ridx] = crect;
            //        }
            //    }
            //}

            return rectlist;
        }

        private List<OpenCvSharp.Rect> GetCoorBond2x1(Mat xyenhance4x, Mat edged, int widthlow, int widthhigh)
        {
            var ret = new List<OpenCvSharp.Rect>();

            var outmat = new Mat();
            var ids = OutputArray.Create(outmat);
            var cons = new Mat[] { };
            Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            var idx1 = 0;
            foreach (var item in cons)
            {
                var crect = Cv2.BoundingRect(item);
                if (crect.Width >= widthlow && crect.Width <= widthhigh && crect.Height > 65)
                {
                    ret.Add(crect);
                    //Cv2.Rectangle(xyenhance4x, crect, new Scalar(0, 255, 0));
                    //using (new Window("xyenhance4rg" + idx1, xyenhance4x))
                    //{
                    //    Cv2.WaitKey();
                    //}
                    //idx1++;

                }//end if
            }

            return ret;
        }

        private static List<List<double>> GetCoordPT2X1(Mat edged)
        {
            var ret = new List<List<double>>();

            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(edged, null, out kazeKeyPoints, kazeDescriptors);

            var allpt = kazeKeyPoints.ToList();

            var wptlist = new List<KeyPoint>();
            for (var idx = 15; idx < edged.Width;)
            {
                var ylist = new List<double>();
                var wlist = new List<KeyPoint>();
                foreach (var pt in allpt)
                {
                    if (pt.Pt.X >= (idx - 15) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        ylist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 16 && (ylist.Max() - ylist.Min()) > 0.6 * edged.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 15;
            }


            var hptlist = new List<KeyPoint>();
            for (var idx = 15; idx < edged.Height;)
            {
                var xlist = new List<double>();
                var hlist = new List<KeyPoint>();
                foreach (var pt in wptlist)
                {
                    if (pt.Pt.Y >= (idx - 15) && pt.Pt.Y < idx)
                    {
                        hlist.Add(pt);
                        xlist.Add(pt.Pt.X);
                    }
                }

                if (hlist.Count > 16 && (xlist.Max() - xlist.Min()) > 0.65 * edged.Width)

                { hptlist.AddRange(hlist); }
                idx = idx + 15;
            }

            if (hptlist.Count() == 0)
            {
                return ret;
            }

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(edged, hptlist.ToArray(), dstKaze);

            using (new Window("dstKaze", dstKaze))
            {
                Cv2.WaitKey();
            }

            var xxlist = new List<double>();
            var yylist = new List<double>();
            foreach (var pt in hptlist)
            {
                xxlist.Add(pt.Pt.X);
                yylist.Add(pt.Pt.Y);
            }
            ret.Add(xxlist);
            ret.Add(yylist);

            return ret;
        }

        private static List<List<double>> GetDetectPoint(Mat mat)
        {
            var ret = new List<List<double>>();

            var xyenhance = new Mat();
            Cv2.DetailEnhance(mat, xyenhance);

            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(xyenhance, null, out kazeKeyPoints, kazeDescriptors);

            var wptlist = new List<KeyPoint>();
            for (var idx = 20; idx < mat.Width;)
            {
                var yhlist = new List<double>();
                var wlist = new List<KeyPoint>();
                foreach (var pt in kazeKeyPoints)
                {
                    if (pt.Pt.X >= (idx - 20) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        yhlist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 10 && (yhlist.Max() - yhlist.Min()) > 0.3 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 20;
            }

            var hptlist = new List<KeyPoint>();
            for (var idx = 20; idx < mat.Height;)
            {
                var xwlist = new List<double>();
                var wlist = new List<KeyPoint>();
                foreach (var pt in wptlist)
                {
                    if (pt.Pt.Y >= (idx - 20) && pt.Pt.Y < idx)
                    {
                        wlist.Add(pt);
                        xwlist.Add(pt.Pt.X);
                    }
                }

                if (wlist.Count >= 2 && (xwlist.Max() - xwlist.Min()) > 0.3 * mat.Width)
                { hptlist.AddRange(wlist); }
                idx = idx + 20;
            }

            var xlist = new List<double>();
            var ylist = new List<double>();
            foreach (var pt in hptlist)
            {
                xlist.Add(pt.Pt.X);
                ylist.Add(pt.Pt.Y);
            }
            ret.Add(xlist);
            ret.Add(ylist);

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(mat, wptlist, dstKaze);

            using (new Window("dstKazexx", dstKaze))
            {
                Cv2.WaitKey();
            }

            return ret;
        }


        private bool XUniformity(List<int> xlist, int widthlow, int widthhigh)
        {
            if (xlist.Count == 4)
            {
                for (var idx = 1; idx < xlist.Count; idx++)
                {
                    var delta = xlist[idx] - xlist[idx - 1];
                    if (delta >= widthlow && delta < (widthhigh + 10))
                    { }
                    else
                    { return false; }
                }
                return true;
            }
            else
            { return false; }

        }

        private List<int> GetPossibleXList2x1(Mat edged, int widthlow, int widthhigh)
        {
            var outmat = new Mat();
            var ids = OutputArray.Create(outmat);
            var cons = new Mat[] { };
            Cv2.FindContours(edged, out cons, ids, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var cwlist = new List<int>();
            var y0list = new List<int>();
            var y1list = new List<int>();

            var idx1 = 0;
            foreach (var item in cons)
            {
                idx1++;
                var crect = Cv2.BoundingRect(item);
                if (crect.Width >= widthlow && crect.Width <= widthhigh)
                {
                    if (crect.Y > (edged.Height / 2 - 30))
                    {
                        y1list.Add(crect.Y);
                        cwlist.Add(crect.X);

                        //var mat = edged.SubMat(crect);
                        //using (new Window("edged" + idx1, mat))
                        //{
                        //    Cv2.WaitKey();
                        //}
                    }
                    else
                    { y0list.Add(crect.Y); }
                }//end if
            }

            cwlist.Sort();

            if (XUniformity(cwlist, widthlow, widthhigh))
            { }
            else
            {
                cwlist.Clear();
                y0list.Clear();
                y1list.Clear();

                for (var idx = 0; idx < 4; idx++)
                {
                    cwlist.Add(-1);
                }

                idx1 = 0;
                foreach (var item in cons)
                {
                    idx1++;
                    var crect = Cv2.BoundingRect(item);
                    if (crect.Width >= widthlow && crect.Width <= widthhigh)
                    {
                        if (crect.Y > (edged.Height / 2 - 30))
                        {
                            y1list.Add(crect.Y);
                            if (crect.X < 45)
                            {
                                cwlist[0] = crect.X;
                                cwlist[1] = crect.X + crect.Width;
                            }
                            else if (crect.X >= 50 && crect.X < 100)
                            {
                                cwlist[1] = crect.X;
                                cwlist[2] = crect.X + crect.Width;
                            }
                            else if (crect.X > 105 && crect.X < 160)
                            {
                                cwlist[2] = crect.X;
                                cwlist[3] = crect.X + crect.Width;
                            }
                            else if (crect.X > 165 && crect.X < 210)
                            { cwlist[3] = crect.X; }
                        }
                        else
                        { y0list.Add(crect.Y); }


                    }//end if
                }//end foreach
            }

            if (cwlist[3] == -1)
            { cwlist[3] = edged.Width - 60; }
            if (cwlist[2] == -1)
            { cwlist[2] = cwlist[3] - 60; }
            if (cwlist[1] == -1)
            { cwlist[1] = cwlist[2] - 60; }
            if (cwlist[0] == -1)
            { cwlist[0] = 5; }

            if (y0list.Count > 0)
            { cwlist.Add((int)y0list.Average()); }
            else
            { cwlist.Add(0); }

            if (y1list.Count > 0)
            { cwlist.Add((int)y1list.Average()); }
            else
            { cwlist.Add((int)(edged.Height * 0.55)); }

            return cwlist;
        }
        private List<OpenCvSharp.Rect> FindXYRect5X1_(Mat src,Mat blurred,Mat srccolor, bool cflag, int heighlow, int heighhigh, int arealow, int areahigh)
        {
            var ret = new List<OpenCvSharp.Rect>();
            var edged = new Mat();
            Cv2.Canny(blurred, edged, 50, 200, 3, cflag);

            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            var outmat = new Mat();
            var ids = OutputArray.Create(outmat);
            var cons = new Mat[] { };
            Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            var conslist = cons.ToList();
            //conslist.Sort(delegate (Mat obj1, Mat obj2)
            //{
            //    return Cv2.ContourArea(obj2).CompareTo(Cv2.ContourArea(obj1));
            //});

            var idx = 0;
            foreach (var item in conslist)
            {
                idx++;

                var rect = Cv2.BoundingRect(item);
                var whrate = (double)rect.Width / (double)rect.Height;
                var a = rect.Width * rect.Height;
                if (rect.Height >= heighlow && rect.Height <= heighhigh
                    && whrate >= 4.5 && whrate < 6.8 && a < 8000)

                {
                    //Cv2.Rectangle(srccolor, rect, new Scalar(0, 255, 0),3);
                    //using (new Window("xyenhance4", srccolor))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    var xymat = src.SubMat(rect);

                    //using (new Window("xymat" + idx, xymat))
                    //{
                    //    Cv2.WaitKey();
                    //}

                    if (ret.Count > 0)
                    {
                        if (a > ret[0].Width * ret[0].Height)
                        {
                            ret.Clear();
                            ret.Add(rect);
                        }
                    }
                    else
                    { ret.Add(rect); }
                }
            }

            return ret;
        }


        private List<OpenCvSharp.Rect> FindXYRect5X1(string file, double angle, int heighlow, int heighhigh, int arealow, int areahigh)
        {
            var ret = new List<OpenCvSharp.Rect>();
            var srccolor = Cv2.ImRead(file, ImreadModes.Color);

            if (angle >= 0.7 && angle <= 359.3)
            {
                var center = new Point2f(srccolor.Width / 2, srccolor.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, angle, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(srccolor, outxymat, m, new Size(srccolor.Width, srccolor.Height));
                srccolor = outxymat;
            }

            Mat src = new Mat();
            Cv2.CvtColor(srccolor, src, ColorConversionCodes.BGR2GRAY);

            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoising(src, denoisemat, 10, 7, 21);

            var blurred = new Mat();
            Cv2.GaussianBlur(denoisemat, blurred, new Size(5, 5), 0);

            var truerect = FindXYRect5X1_(src, blurred, srccolor, true, heighlow,  heighhigh,  arealow,  areahigh);
            var falserect = FindXYRect5X1_(src, blurred, srccolor, false, heighlow, heighhigh, arealow, areahigh);

            if (truerect.Count > 0 && falserect.Count > 0)
            {
                if (truerect[0].Width * truerect[0].Height >= falserect[0].Width * falserect[0].Height)
                {  ret.AddRange(truerect); }
                else
                { ret.AddRange(falserect); }
            }
            else if (truerect.Count > 0)
            { ret.AddRange(truerect); }
            else
            { ret.AddRange(falserect); }


            if (ret.Count > 0)
            {
                var charmat = srccolor.SubMat(ret[0]);
                Cv2.DetailEnhance(charmat, charmat);
                var charmat4x = new Mat();
                Cv2.Resize(charmat, charmat4x, new Size(charmat.Width * 4, charmat.Height * 4));
                Cv2.DetailEnhance(charmat4x, charmat4x);

                var kaze = KAZE.Create();
                var kazeDescriptors = new Mat();
                KeyPoint[] kazeKeyPoints = null;
                kaze.DetectAndCompute(charmat4x, null, out kazeKeyPoints, kazeDescriptors);
                var hptlist = new List<KeyPoint>();
                var cl = 0.3 * charmat4x.Height;
                var ch = 0.7 * charmat4x.Height;
                var rl = 60;
                var rlh = charmat4x.Width * 0.3;
                var rhl = charmat4x.Width * 0.7;
                var rh = charmat4x.Width - 60;


                foreach (var pt in kazeKeyPoints)
                {
                    if (pt.Pt.Y >= cl && pt.Pt.Y <= ch
                        && ((pt.Pt.X >= rl && pt.Pt.X <= rlh) || (pt.Pt.X >= rhl && pt.Pt.X <= rh)))
                    {
                        hptlist.Add(pt);
                    }
                }

                if (hptlist.Count < 140)
                {
                    return new List<Rect>();
                }
            }


            src.Dispose();
            return ret;         
        }

        private List<int> UniqX(List<int> xlist,int imgw,int avgw, int widthlow, int widthhigh)
        {
            var ret = new List<int>();
            if (xlist.Count < 3) { return ret; }

            ret.Add(xlist[0]);
            for (var idx = 1; idx < xlist.Count; idx++)
            {
                var delta = xlist[idx] - xlist[idx-1];
                if (delta > widthlow)
                { ret.Add(xlist[idx]); }
            }

            if (ret.Count == 8)
            { return ret; }
            else
            {

                var passiblexlist = new List<int>();
                for (var idx = 0; idx < 8; idx++)
                { passiblexlist.Add(-1); }

                var xhigh = (int)(imgw * 0.666);
                var step = avgw + 2;

                var leftready = false;
                var rightready = false;

                foreach (var x in ret)
                {
                    if (x < (int)(step) && !leftready)
                    {
                        passiblexlist[0] = x;
                        passiblexlist[1] = passiblexlist[0] + avgw + 4;
                        passiblexlist[2] = passiblexlist[1] + avgw + 6;
                        passiblexlist[3] = passiblexlist[2] + avgw + 8;
                        leftready = true;
                    }
                    else if (x > (3 * step + 8) && x < (4 * step + 20) && !leftready)
                    {
                        passiblexlist[3] = x;
                        passiblexlist[2] = passiblexlist[3] - avgw - 5;
                        passiblexlist[1] = passiblexlist[2] - avgw - 6;
                        passiblexlist[0] = passiblexlist[1] - avgw - 8;
                        leftready = true;
                    }
                    else if (x > (step + 8) && x < (2 * step - 10) && !leftready)
                    {
                        passiblexlist[1] = x;
                        passiblexlist[2] = passiblexlist[1] + avgw + 5;
                        passiblexlist[3] = passiblexlist[2] + avgw + 8;
                        if (passiblexlist[0] == -1)
                        { passiblexlist[0] = passiblexlist[1] - avgw - 5; }
                    }
                    else if (x > (2 * step + 8) && x < 3 * step && !leftready)
                    {
                        passiblexlist[2] = x;
                        passiblexlist[3] = passiblexlist[2] + avgw + 5;
                        if (passiblexlist[1] == -1)
                        { passiblexlist[1] = passiblexlist[2] - avgw - 5; }
                        if (passiblexlist[0] == -1)
                        { passiblexlist[0] = passiblexlist[1] - avgw - 8; }
                    }
                    else if ((int)Math.Abs(x - xhigh) < (int)(step) && !rightready)
                    {
                        passiblexlist[4] = x;
                        passiblexlist[5] = passiblexlist[4] + avgw + 4;
                        passiblexlist[6] = passiblexlist[5] + avgw + 6;
                        passiblexlist[7] = passiblexlist[6] + avgw + 8;
                        rightready = true;
                    }
                    else if (x > imgw - step * 1.5 && !rightready)
                    {
                        passiblexlist[7] = x;
                        passiblexlist[6] = passiblexlist[7] - avgw - 5;
                        passiblexlist[5] = passiblexlist[6] - avgw - 6;
                        passiblexlist[4] = passiblexlist[5] - avgw - 8;
                        rightready = true;
                    }
                    else if (x > imgw - step * 3 && x < imgw - step * 2 && !rightready)
                    {
                        passiblexlist[6] = x;
                        passiblexlist[7] = passiblexlist[6] + avgw + 5;
                        if (passiblexlist[5] == -1)
                        { passiblexlist[5] = passiblexlist[6] - avgw - 5; }
                        if (passiblexlist[4] == -1)
                        { passiblexlist[4] = passiblexlist[5] - avgw - 6; }
                    }
                    else if ((int)Math.Abs(x - xhigh) > (int)(1.1 * step) && (int)Math.Abs(x - xhigh) < (int)(2.1 * step) && !rightready)
                    {
                        passiblexlist[5] = x;
                        passiblexlist[6] = passiblexlist[5] + avgw + 5;
                        passiblexlist[7] = passiblexlist[6] + avgw + 6;
                        if (passiblexlist[4] == -1)
                        { passiblexlist[4] = passiblexlist[5] - avgw - 5; }
                    }

                }//end foreach

                for (var idx = 1; idx < passiblexlist.Count; idx++)
                {
                    var delta = passiblexlist[idx] - passiblexlist[idx - 1];
                    if (delta < widthlow)
                    {
                        passiblexlist.Clear();
                        return passiblexlist;
                    }

                    if (idx != 4 && delta > widthhigh) {
                        passiblexlist.Clear();
                        return passiblexlist;
                    }
                 }

                foreach (var x in passiblexlist)
                {
                    if (x == -1)
                    {
                        passiblexlist.Clear();
                        return passiblexlist;
                    }
                }

                if (passiblexlist[0] < 0)
                {
                    passiblexlist.Clear();
                    return passiblexlist;
                }

                if (Math.Abs(passiblexlist[7] - imgw) <= 4)
                {
                    passiblexlist[7] = imgw - avgw - 4;
                    passiblexlist[6] = passiblexlist[7] - avgw - 5;
                    passiblexlist[5] = passiblexlist[6] - avgw - 6;
                    passiblexlist[4] = passiblexlist[5] - avgw - 8;
                }

                return passiblexlist;
            }
        }


        //private List<int> GetPossibleXList5X1(Mat edged, int heighlow, int heighhigh, int widthlow, int widthhigh, Mat xyenhance4)
        //{
        //    var outmat = new Mat();
        //    var ids = OutputArray.Create(outmat);
        //    var cons = new Mat[] { };
        //    Cv2.FindContours(edged, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        //    var xlow = (int)(edged.Width*0.333 - 20);
        //    var xhigh = (int)(edged.Width * 0.666 - 10);

        //    var cwlist = new List<int>();
        //    var y0list = new List<int>();
        //    var y1list = new List<int>();
        //    var wavglist = new List<int>();

        //    var backxlist = new List<KeyValuePair<int,int>>();

        //    var idx1 = 0;
        //    foreach (var item in cons)
        //    {
        //        idx1++;

        //        var crect = Cv2.BoundingRect(item);

        //        //if (crect.Width > 20 && crect.Height > 20)
        //        //{
        //        //    Cv2.Rectangle(xyenhance4, crect, new Scalar(0, 255, 0));
        //        //    using (new Window("xyenhance4" + idx1, xyenhance4))
        //        //    {
        //        //        Cv2.WaitKey();
        //        //    }
        //        //}

        //        if (crect.Width > 25 && crect.Width < 40 && crect.Height > 50 && crect.Height < 90)
        //        {
        //            //Cv2.Rectangle(xyenhance4, crect, new Scalar(255, 0, 0));
        //            //using (new Window("xyenhance4tb" + idx1, xyenhance4))
        //            //{
        //            //    Cv2.WaitKey();
        //            //}

        //            if ((crect.X < xlow || crect.X > xhigh)  && crect.Y > 5)
        //            {                    
        //                backxlist.Add(new KeyValuePair<int, int>(crect.X,crect.Width));
        //            }//end if
        //        }

        //        if ( crect.Width >= widthlow && crect.Width <= widthhigh && crect.Height > 50 && crect.Height < 90)
        //        {
        //            //Cv2.Rectangle(xyenhance4, crect, new Scalar(0, 255, 0));
        //            //using (new Window("xyenhance4rg" + idx1, xyenhance4))
        //            //{
        //            //    Cv2.WaitKey();
        //            //}

        //            if ((crect.X < xlow || crect.X > xhigh) && crect.X > 5 &&  crect.Y >= 4)
        //            {
        //                var mat = edged.SubMat(crect);

        //                wavglist.Add(crect.Width);
        //                cwlist.Add(crect.X);
        //                y0list.Add(crect.Y);
        //                y1list.Add(crect.Height);
        //            }

        //        }//end if
        //    }//end foreach


        //    cwlist.Sort();
        //    var wavg = 0;
        //    if (wavglist.Count > 0)
        //    { wavg = (int)wavglist.Average(); }

        //    var retlist = UniqX(cwlist,edged.Width, wavg, widthlow,widthhigh);
        //    if (retlist.Count == 0)
        //    {
        //        var ncwlist = new List<int>();
        //        ncwlist.AddRange(cwlist);
        //        foreach (var kv in backxlist)
        //        {
        //            var bx = kv.Key - (wavg - kv.Value) / 2 - 2;
        //            if (bx > 0)
        //            { ncwlist.Add(bx); }
        //        }
        //        ncwlist.Sort();
        //        retlist = UniqX(ncwlist, edged.Width, wavg, widthlow, widthhigh);
        //    }


        //    if (retlist.Count == 8)
        //    {
        //        retlist.Add((int)y0list.Average());
        //        retlist.Add((int)y1list.Max()+2);
        //    }

        //    return retlist;
        //}



        //private List<Mat> CutCharRect5X1(string imgpath, OpenCvSharp.Rect xyrect, int heighlow, int heighhigh, int widthlow, int widthhigh)
        //{
        //    var cmatlist = new List<Mat>();

        //    Mat src = Cv2.ImRead(imgpath, ImreadModes.Color);
        //    var xymat = src.SubMat(xyrect);

        //    var availableimgpt = GetDetectPoint(src);
        //    //var srcmidy = src.Height / 2;
        //    var srcmidy = (availableimgpt[1].Max() + availableimgpt[1].Min()) / 2;

        //    if (xyrect.Y+xyrect.Height > srcmidy)
        //    {
        //        var center = new Point2f(xymat.Width / 2, xymat.Height / 2);
        //        var m = Cv2.GetRotationMatrix2D(center, 180, 1);
        //        var outxymat = new Mat();
        //        Cv2.WarpAffine(xymat, outxymat, m, new Size(xymat.Width, xymat.Height));
        //        xymat = outxymat;
        //    }

        //    var xyenhance = new Mat();
        //    Cv2.DetailEnhance(xymat, xyenhance);

        //    var xyenhance4 = new Mat();
        //        Cv2.Resize(xyenhance, xyenhance4, new Size(xyenhance.Width * 4, xyenhance.Height * 4));


        //    var xyenhgray = new Mat();
        //    Cv2.CvtColor(xyenhance, xyenhgray, ColorConversionCodes.BGR2GRAY);

        //    Cv2.Resize(xyenhgray, xyenhgray, new Size(xyenhgray.Width * 4, xyenhgray.Height * 4));

        //    //using (new Window("xyenhgray", xyenhgray))
        //    //{
        //    //    Cv2.WaitKey();
        //    //}

        //    var xyenhance4list = new List<Mat>();

        //    var subx = new List<int>();
        //    subx.Add(5);subx.Add(15);
        //    subx.Add(25);subx.Add(30);
        //    var xyenhgraylist = new List<Mat>();
        //    foreach (var x in subx)
        //    {
        //        var tmpmat = new Mat();
        //        tmpmat = xyenhgray.SubMat(x, xyenhgray.Rows - x, x, xyenhgray.Cols - x);
        //        xyenhgraylist.Add(tmpmat);

        //        xyenhance4list.Add( xyenhance4.SubMat(x, xyenhance4.Rows - x, x, xyenhance4.Cols - x));
        //    }

        //    var idx = 0;
        //    foreach(var tempmat in xyenhgraylist)
        //    {
        //        var blurred = new Mat();
        //        Cv2.GaussianBlur(tempmat, blurred, new Size(5, 5), 0);

        //        //using (new Window("blurred", blurred))
        //        //{
        //        //    Cv2.WaitKey();
        //        //}


        //        var edged = new Mat();
        //        Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 11, 2);
        //        using (new Window("edged", edged))
        //        {
        //            Cv2.WaitKey();
        //        }

        //        //cmatlist.Add(xyenhance);
        //        //var crectlist = Get5x1Rect(blurred, edged, xyenhance4list[idx]);
        //        //foreach (var rect in crectlist)
        //        //{
        //        //    cmatlist.Add(edged.SubMat(rect));
        //        //}

        //        //return cmatlist;

        //        var possxlist = GetPossibleXList5X1(edged, heighlow, heighhigh, 40, 67, xyenhance4list[idx]);
        //        idx = idx + 1;

        //        if (possxlist.Count > 0)
        //        {
        //            var ewidth1 = possxlist[3] * 2 - possxlist[2] + 3;
        //            var ewidth2 = possxlist[7] * 2 - possxlist[6];
        //            if (ewidth2 > edged.Width)
        //            { ewidth2 = edged.Width; }

        //            var eheight0 = possxlist[8];
        //            var eheight1 = possxlist[8] + possxlist[9];
        //            if (eheight1 > edged.Height)
        //            { eheight1 = edged.Height; }

        //            cmatlist.Add(xyenhance);

        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[0] > 0 ? possxlist[0] : 0, possxlist[1]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[1], possxlist[2]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[2], possxlist[3]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[3], ewidth1));

        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[4], possxlist[5]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[5], possxlist[6]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[6], possxlist[7]));
        //            cmatlist.Add(edged.SubMat(eheight0, eheight1, possxlist[7], ewidth2));
        //            return cmatlist;
        //        }
        //    }


        //    return cmatlist;
        //}

        public static List<Rect> GetNew5x1Rect(Mat edged, Mat xyenhance4x)
        {
            var hl = GetHeighLow(edged);
            var hh = GetHeighHigh(edged);
            if (hl < 6)
            { hl = 10; }
            if (hh > edged.Height - 6)
            { hh = edged.Height - 10; }

            var dcl = (int)(hl + (hh - hl) * 0.1);
            var dch = (int)(hh - (hh - hl) * 0.1);
            var xxh = GetXXHigh(edged, dcl, dch);
            var yxl = GetYXLow(edged, dcl, dch);
            if (xxh == -1 || yxl == -1)
            {
                var xlist = GetCoordWidthPT1(xyenhance4x, edged);
                var xmid = (xlist.Max() + xlist.Min()) / 2;
                var xcxlist = new List<double>();
                var ycxlist = new List<double>();
                foreach (var x in xlist)
                {
                    if (x < xmid)
                    { xcxlist.Add(x); }
                    else
                    { ycxlist.Add(x); }
                }
                xxh = (int)xcxlist.Max() + 2;
                yxl = (int)ycxlist.Min() - 2;
            }

            var rectlist = new List<Rect>();

            var xxlist = GetXSplitList(edged, xxh, hl, hh);
            var flist = (List<int>)xxlist[0];
            var slist = (List<int>)xxlist[1];
            var y = hl - 2;
            var h = hh - hl + 4;

            if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                var left = slist[2] - fntw - 3;
                if (left < 0) { left = 1; }
                rectlist.Add(new Rect(left, y, fntw + 1, h));
                rectlist.Add(new Rect(slist[2] - 3, y, slist[1] - slist[2], h));
                rectlist.Add(new Rect(slist[1] - 3, y, slist[0] - slist[1], h));
                rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 2, h));
            }
            else if (slist.Count == 2)
            {
                var fntw = (int)flist.Average();
                var left = slist[1] - 2 * fntw - 4;
                if (left < 0) { left = 1; }
                rectlist.Add(new Rect(left, y, fntw + 1, h));
                rectlist.Add(new Rect(slist[1] - fntw - 3, y, fntw + 1, h));
                rectlist.Add(new Rect(slist[1] - 3, y, slist[0] - slist[1], h));
                rectlist.Add(new Rect(slist[0] - 3, y, xxh - slist[0] + 2, h));
            }
            else
            {
                if ((int)xxh - 226 > 0)
                { rectlist.Add(new Rect(xxh - 226, y, 48, h)); }
                else
                { rectlist.Add(new Rect(0, y, 48, h)); }
                rectlist.Add(new Rect(xxh - 164, y, 48, h));
                rectlist.Add(new Rect(xxh - 110, y, 48, h));
                rectlist.Add(new Rect(xxh - 55, y, 48, h));
            }

            var yxlist = GetYSplitList(edged, yxl, hl, hh);
            flist = (List<int>)yxlist[0];
            slist = (List<int>)yxlist[1];
            if (slist.Count == 4)
            {
                rectlist.Add(new Rect(yxl - 1, y, slist[0] - yxl + 2, h));
                rectlist.Add(new Rect(slist[0] + 3, y, slist[1] - slist[0], h));
                rectlist.Add(new Rect(slist[1] + 3, y, slist[2] - slist[1], h));
                rectlist.Add(new Rect(slist[2] + 3, y, slist[3] - slist[2], h));
            }
            else if (slist.Count == 3)
            {
                var fntw = (int)flist.Average();
                rectlist.Add(new Rect(yxl - 1, y, slist[0] - yxl + 2, h));
                rectlist.Add(new Rect(slist[0] + 3, y, slist[1] - slist[0], h));
                rectlist.Add(new Rect(slist[1] + 3, y, slist[2] - slist[1], h));
                var left = slist[2] + 3;
                if (left + fntw > edged.Width)
                { left = edged.Width - fntw - 2; }
                rectlist.Add(new Rect(left, y, fntw, h));
            }
            else if (slist.Count == 2)
            {
                var fntw = (int)flist.Average();
                rectlist.Add(new Rect(yxl - 1, y, slist[0] - yxl + 2, h));
                rectlist.Add(new Rect(slist[0] + 3, y, slist[1] - slist[0], h));
                rectlist.Add(new Rect(slist[1] + 3, y, fntw+2, h));
                var left = slist[1] + fntw + 3;
                if (left + fntw+4 > edged.Width)
                { left = edged.Width - fntw - 4; }
                rectlist.Add(new Rect(left, y, fntw+2, h));
            }
            else
            {
                rectlist.Add(new Rect(yxl - 2, y, 48, h));
                rectlist.Add(new Rect(yxl + 53, y, 48, h));
                rectlist.Add(new Rect(yxl + 110, y, 48, h));
                if ((yxl + 211) >= (edged.Cols - 1))
                { rectlist.Add(new Rect(yxl + 161, y, edged.Cols - yxl - 161, h)); }
                else
                { rectlist.Add(new Rect(yxl + 161, y, 50, h)); }
            }
            return rectlist;
        }

        public static int GetHeighLow(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.20);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.79);

            var xhl = 0;
            var yhl = 0;
            var ymidx = (int)(edged.Height * 0.5);
            for (var idx = ymidx; idx > 10; idx = idx - 2)
            {
                if (xhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhl = idx;
                    }
                }

                if (yhl == 0)
                {
                    var snapmat = edged.SubMat(idx - 2, idx, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhl = idx;
                    }
                }
            }

            var hl = xhl;
            if (yhl > hl)
            { hl = yhl; }

            return hl;
        }

        public static int GetHeighHigh(Mat edged)
        {
            var cheighxl = (int)(edged.Width * 0.20);
            var cheighxh = (int)(edged.Width * 0.33);
            var cheighyl = (int)(edged.Width * 0.66);
            var cheighyh = (int)(edged.Width * 0.79);

            var xhh = 0;
            var yhh = 0;
            var ymidx = (int)(edged.Height * 0.5);
            for (var idx = ymidx; idx < edged.Height - 10; idx = idx + 2)
            {
                if (xhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighxl, cheighxh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        xhh = idx;
                    }
                }

                if (yhh == 0)
                {
                    var snapmat = edged.SubMat(idx, idx + 2, cheighyl, cheighyh);
                    var cnt = snapmat.CountNonZero();
                    if (cnt < 3)
                    {
                        yhh = idx;
                    }
                }
            }

            var hh = 0;
            if (xhh > ymidx && yhh > ymidx)
            {
                if (yhh < xhh)
                { hh = yhh; }
                else
                { hh = xhh; }
            }
            else if (xhh > ymidx)
            { hh = xhh; }
            else if (yhh > ymidx)
            { hh = yhh; }
            else
            { hh = edged.Height - 10; }
            return hh;
        }

        public static int GetXXHigh(Mat edged, int dcl, int dch)
        {
            var wml = (int)(edged.Width * 0.35);

            for (var idx = wml; idx > wml / 2; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    return idx;
                }
            }

            return -1;
        }

        public static int GetYXLow(Mat edged, int dcl, int dch)
        {
            var wml = (int)(edged.Width * 0.35);
            var wmh = (int)(edged.Width * 0.65);

            for (var idx = wmh; idx < (wmh + wml / 2); idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt > 3)
                {
                    return idx;
                }
            }
            return -1;
        }

        public static int GetXDirectSplit(Mat edged, int start, int end, int dcl, int dch)
        {
            for (var idx = start; idx > end; idx = idx - 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx - 2, idx);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    return idx;
                }
            }
            return -1;
        }

        public static int GetYDirectSplit(Mat edged, int start, int end, int dcl, int dch)
        {
            for (var idx = start; idx < end; idx = idx + 2)
            {
                var snapmat = edged.SubMat(dcl, dch, idx, idx + 2);
                var cnt = snapmat.CountNonZero();
                if (cnt < 2)
                {
                    return idx;
                }
            }
            return -1;
        }

        public static List<object> GetXSplitList(Mat edged, int xxh, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spx1 = GetXDirectSplit(edged, xxh - 20, xxh - 20 - fntw, hl, hh);
            if (spx1 == -1) { return ret; }
            fntw = xxh - spx1 + 1;
            if (fntw >= 18 && fntw < 40)
            { spx1 = xxh - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx1);

            var spx2 = GetXDirectSplit(edged, spx1 - 21, spx1 - 21 - fntw, hl, hh);
            if (spx2 == -1) { return ret; }
            fntw = spx1 - spx2;
            if (fntw >= 18 && fntw < 40)
            { spx2 = spx1 - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx2);

            var spx3 = GetXDirectSplit(edged, spx2 - 21, spx2 - 21 - fntw, hl, hh);
            if (spx3 == -1) { return ret; }
            fntw = spx2 - spx3;
            if (fntw >= 18 && fntw < 40)
            { spx3 = spx2 - offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spx3);

            return ret;
        }
        public static List<object> GetYSplitList(Mat edged, int yxl, int hl, int hh)
        {
            var offset = 50;
            var ret = new List<object>();
            var flist = new List<int>();
            var slist = new List<int>();
            ret.Add(flist);
            ret.Add(slist);

            var fntw = (int)(edged.Width * 0.333 * 0.25);

            var spy1 = GetYDirectSplit(edged, yxl + 20, yxl + 20 + fntw, hl, hh);
            if (spy1 == -1) { return ret; }
            fntw = spy1 - yxl + 1;
            if (fntw >= 18 && fntw < 40)
            { spy1 = yxl + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy1);

            var spy2 = GetYDirectSplit(edged, spy1 + 21, spy1 + 21 + fntw, hl, hh);
            if (spy2 == -1) { return ret; }
            fntw = spy2 - spy1 + 1;
            if (fntw >= 18 && fntw < 40)
            { spy2 = spy1 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy2);

            var spy3 = GetYDirectSplit(edged, spy2 + 20, spy2 + 20 + fntw, hl, hh);
            if (spy3 == -1) { return ret; }
            fntw = spy3 - spy2 + 1;
            if (fntw >= 18 && fntw < 40)
            { spy3 = spy2 + offset; fntw = offset; }
            flist.Add(fntw); slist.Add(spy3);

            var spy4 = GetYDirectSplit(edged, spy3 + 20, edged.Width - 10, (int)(hl + 0.1 * (hh - hl)), (int)(hh - 0.1 * (hh - hl)));
            if (spy4 == -1) { return ret; }
            fntw = spy4 - spy3 + 1;
            if (fntw < 45)
            { return ret; }
            flist.Add(fntw); slist.Add(spy4);

            return ret;
        }



        private List<Mat> CutCharRect5X1xxx(string imgpath,double angle, OpenCvSharp.Rect xyrect, int heighlow, int heighhigh, int widthlow, int widthhigh)
        {
            var cmatlist = new List<Mat>();

            Mat src = Cv2.ImRead(imgpath, ImreadModes.Color);
            if (angle >= 0.7 && angle <= 359.3)
            {
                var center = new Point2f(src.Width / 2, src.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, angle, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(src, outxymat, m, new Size(src.Width, src.Height));
                src = outxymat;
            }

            var xymat = src.SubMat(xyrect);

            var availableimgpt = GetDetectPoint(src);
            //var srcmidy = src.Height / 2;
            var srcmidy = (availableimgpt[1].Max() + availableimgpt[1].Min()) / 2;

            if (xyrect.Y + xyrect.Height > srcmidy)
            {
                var center = new Point2f(xymat.Width / 2, xymat.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(xymat, outxymat, m, new Size(xymat.Width, xymat.Height));
                xymat = outxymat;
            }


            var xyenhance = new Mat();
            Cv2.DetailEnhance(xymat, xyenhance);

            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 4, xyenhance.Height * 4));

            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);


            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);
            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            var rectlist = GetNew5x1Rect(edged, xyenhance4x);

            cmatlist.Add(xyenhance);

            foreach (var rect in rectlist)
            {
                if (rect.X < 0 || rect.Y < 0
                || ((rect.X + rect.Width) > edged.Width)
                || ((rect.Y + rect.Height) > edged.Height))
                {
                    cmatlist.Clear();
                    return cmatlist;
                }

                cmatlist.Add(edged.SubMat(rect));
            }

            return cmatlist;

        }



        private List<Mat> CutCharRect6X1xxx(string imgpath, double angle, OpenCvSharp.Rect xyrect, int heighlow, int heighhigh, int widthlow, int widthhigh)
        {
            var cmatlist = new List<Mat>();

            Mat src = Cv2.ImRead(imgpath, ImreadModes.Color);
            if (angle >= 0.7 && angle <= 359.3)
            {
                var center = new Point2f(src.Width / 2, src.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, angle, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(src, outxymat, m, new Size(src.Width, src.Height));
                src = outxymat;
            }

            var xymat = src.SubMat(xyrect);

            var availableimgpt = GetDetectPoint(src);
            //var srcmidy = src.Height / 2;
            var srcmidy = (availableimgpt[1].Max() + availableimgpt[1].Min()) / 2;

            if (xyrect.Y + xyrect.Height > srcmidy)
            {
                var center = new Point2f(xymat.Width / 2, xymat.Height / 2);
                var m = Cv2.GetRotationMatrix2D(center, 180, 1);
                var outxymat = new Mat();
                Cv2.WarpAffine(xymat, outxymat, m, new Size(xymat.Width, xymat.Height));
                xymat = outxymat;
            }

            var xyenhance = new Mat();
            Cv2.DetailEnhance(xymat, xyenhance);



            var xyenhance4x = new Mat();
            Cv2.Resize(xyenhance, xyenhance4x, new Size(xyenhance.Width * 6, xyenhance.Height * 6));

            Cv2.DetailEnhance(xyenhance4x, xyenhance4x);

            var xyenhgray = new Mat();
            var denoisemat = new Mat();
            Cv2.FastNlMeansDenoisingColored(xyenhance4x, denoisemat, 10, 10, 7, 21);
            Cv2.CvtColor(denoisemat, xyenhgray, ColorConversionCodes.BGR2GRAY);


            var blurred = new Mat();
            Cv2.GaussianBlur(xyenhgray, blurred, new Size(5, 5), 0);

            //using (new Window("blurred", blurred))
            //{
            //    Cv2.WaitKey();
            //}

            var edged = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 17, 15);
            using (new Window("edged", edged))
            {
                Cv2.WaitKey();
            }

            cmatlist.Add(xyenhance);
            var crectlist = Get6x1Rect(blurred, edged, xyenhance4x);
            foreach (var rect in crectlist)
            {
                if (rect.X < 0 || rect.Y < 0
                || ((rect.X + rect.Width) > edged.Width)
                || ((rect.Y + rect.Height) > edged.Height))
                {
                    cmatlist.Clear();
                    return cmatlist;
                }

                //var xhen = xyenhance4x.SubMat(rect);
                //using (new Window("xhen", xhen))
                //{
                //    Cv2.WaitKey();
                //}

                cmatlist.Add(edged.SubMat(rect));
            }

            return cmatlist;

        }

        public static List<OpenCvSharp.Rect> Get5x1Rect(Mat blurred, Mat edged, Mat xyenhance4)
        {
            var resizeenhance = new Mat();
            Cv2.DetailEnhance(xyenhance4, resizeenhance);
            var xlist = GetCoordWidthPT1(resizeenhance, edged);
            if (xlist.Count == 0 || (xlist.Max() - xlist.Min()) < 400)
            { return new List<OpenCvSharp.Rect>(); }

            var cbond = GetCoordBond1(blurred, edged, xyenhance4);

            var xmid = (xlist.Max() + xlist.Min()) / 2;
            var xcxlist = new List<double>();
            var ycxlist = new List<double>();
            foreach (var x in xlist)
            {
                if (x < xmid)
                { xcxlist.Add(x); }
                else
                { ycxlist.Add(x); }
            }

            var xcmax = xcxlist.Max() + 3;
            var ycmin = ycxlist.Min() - 3;
            var wdr = 26;

            var ret = new List<OpenCvSharp.Rect>();

            if (cbond.Count > 0)
            {
                cbond.Sort(delegate (OpenCvSharp.Rect o1, OpenCvSharp.Rect o2)
                { return o1.X.CompareTo(o2.X); });
            }

            var filteredbond = new List<OpenCvSharp.Rect>();
            foreach (var item in cbond)
            {

                if (filteredbond.Count == 0)
                {
                    filteredbond.Add(item);
                }
                else
                {
                    var bcnt = filteredbond.Count;
                    if (item.X - filteredbond[bcnt - 1].X > 35)
                    {
                        filteredbond.Add(item);
                    }
                }
            }
 
            if (filteredbond.Count > 1)
            {
                var ylist = new List<int>();
                var hlist = new List<int>();
                foreach (var item in cbond)
                {
                    ylist.Add(item.Y);
                    hlist.Add(item.Height);
                }

                var y0 = (int)ylist.Average() - 2;
                var y1 = (int)hlist.Max() + 1;
                if ((y1 + y0) > edged.Height)
                { y1 = edged.Height - y0 - 1; }

                if ((int)xcmax - 226 > 0)
                {
                    ret.Add(new OpenCvSharp.Rect((int)xcmax - 226, y0, 48, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect(0, y0, 48, y1));
                }

                ret.Add(new OpenCvSharp.Rect((int)xcmax - 164, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 110, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 55, y0, 48, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin-2, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 53, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 110, y0, 48, y1));

                if (((int)ycmin + 211) >= (edged.Cols - 1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, edged.Cols - (int)ycmin - 161, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, 50, y1));
                }


                var changedict = new Dictionary<int, bool>();

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X >= ret[idx].X - wdr) && (item.X <= ret[idx].X + wdr))
                        {
                            var currentrect = new OpenCvSharp.Rect(item.X - 2, ret[idx].Y, item.Width + 4, ret[idx].Height);
                            if (idx == 0)
                            {
                                currentrect = new OpenCvSharp.Rect((item.X - 6) > 0 ? (item.X - 6) : 0, ret[idx].Y, item.Width + 2, ret[idx].Height);
                            }

                            if (idx == 7)
                            {
                                currentrect = new OpenCvSharp.Rect(item.X, ret[idx].Y, (edged.Width - item.Width - 4) > 0 ? (item.Width + 4) : (edged.Width - item.X - 1), ret[idx].Height);
                            }

                            //ret[idx] = currentrect;

                            if (!changedict.ContainsKey(idx) && currentrect.Width < 58)
                            {
                                ret[idx] = currentrect;
                                changedict.Add(idx, true);
                            }
                            break;
                        }//end if
                    }//end foreach
                }//end for

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X >= ret[idx].X - wdr) && (item.X <= ret[idx].X + wdr))
                        {
                            if ((idx >= 0 && idx <= 2) || (idx >= 4 && idx <= 6))
                            {
                                var nextrect = new OpenCvSharp.Rect(item.X + item.Width + 2, ret[idx].Y, item.Width + 4, ret[idx].Height);
                                if (idx + 1 == 7)
                                {
                                    nextrect = new OpenCvSharp.Rect(item.X + item.Width + 4, ret[idx].Y, (edged.Width - item.X - 2 * item.Width - 8) > 0 ? (item.Width + 4) : (edged.Width - item.X - item.Width - 4), ret[idx].Height);
                                }

                                if (!changedict.ContainsKey(idx + 1) && nextrect.Width < 58)
                                {
                                    ret[idx + 1] = nextrect;
                                    changedict.Add(idx + 1, true);
                                }
                            }

                            if ((idx >= 1 && idx <= 3) || (idx >= 5 && idx <= 7))
                            {
                                var nextrect = new OpenCvSharp.Rect((item.X - item.Width - 2) > 0 ? (item.X - item.Width - 2) : 0, ret[idx].Y, item.Width + 4, ret[idx].Height);
                                if (idx - 1 == 0)
                                {
                                    nextrect = new OpenCvSharp.Rect((item.X - item.Width - 6) > 0 ? (item.X - item.Width - 6) : 0, ret[idx].Y, item.Width, ret[idx].Height);
                                }

                                if (!changedict.ContainsKey(idx - 1) && nextrect.Width < 58)
                                {
                                    ret[idx - 1] = nextrect;
                                    changedict.Add(idx - 1, true);
                                }
                            }
                            break;
                        }//end if
                    }//end foreach
                }//end for
            }
            else
            {
                var y0 = 32;
                var y1 = edged.Height - 60;

                if ((int)xcmax - 226 > 0)
                {
                    ret.Add(new OpenCvSharp.Rect((int)xcmax - 226, y0, 48, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect(0, y0, 48, y1));
                }

                ret.Add(new OpenCvSharp.Rect((int)xcmax - 164, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 110, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 55, y0, 48, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin-2, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 53, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 110, y0, 48, y1));

                if (((int)ycmin + 211) >= (edged.Cols - 1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, edged.Cols - (int)ycmin - 161, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, 50, y1));
                }

            }

            return ret;
        }

        public static List<OpenCvSharp.Rect> Get6x1Rect(Mat blurred, Mat edged, Mat xyenhance4)
        {
            var resizeenhance = new Mat();
            Cv2.DetailEnhance(xyenhance4, resizeenhance);
            var xlist = GetCoordWidthPT1(resizeenhance, edged);
            if (xlist.Count == 0 || (xlist.Max() - xlist.Min()) < 400)
            { return new List<OpenCvSharp.Rect>(); }

            var cbond = GetCoordBond2(blurred, edged, xyenhance4);

            var xmid = (xlist.Max() + xlist.Min()) / 2;
            var xcxlist = new List<double>();
            var ycxlist = new List<double>();
            foreach (var x in xlist)
            {
                if (x < xmid)
                { xcxlist.Add(x); }
                else
                { ycxlist.Add(x); }
            }

            var xcmax = xcxlist.Max() + 3;
            var ycmin = ycxlist.Min() - 3;
            var wdr = 26;

            var ret = new List<OpenCvSharp.Rect>();

            if (cbond.Count > 0)
            {
                cbond.Sort(delegate (OpenCvSharp.Rect o1, OpenCvSharp.Rect o2)
                { return o1.X.CompareTo(o2.X); });
            }

            var filteredbond = new List<OpenCvSharp.Rect>();
            foreach (var item in cbond)
            {
                if (filteredbond.Count == 0)
                {
                    filteredbond.Add(item);
                }
                else
                {
                    var bcnt = filteredbond.Count;
                    if (item.X - filteredbond[bcnt - 1].X > 35)
                    {
                        filteredbond.Add(item);
                    }
                }
            }

            if (filteredbond.Count > 1)
            {
                var ylist = new List<int>();
                var hlist = new List<int>();
                foreach (var item in cbond)
                {
                    ylist.Add(item.Y);
                    hlist.Add(item.Height);
                }

                var y0 = (int)ylist.Average() - 2;
                var y1 = (int)hlist.Max() + 1;
                if ((y1 + y0) > edged.Height)
                { y1 = edged.Height - y0 - 1; }

                if ((int)xcmax - 226 > 0)
                {
                    ret.Add(new OpenCvSharp.Rect((int)xcmax - 226, y0, 48, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect(0, y0, 48, y1));
                }

                ret.Add(new OpenCvSharp.Rect((int)xcmax - 164, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 110, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 55, y0, 48, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin - 2, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 53, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 110, y0, 48, y1));

                if (((int)ycmin + 211) >= (edged.Cols - 1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, edged.Cols - (int)ycmin - 161, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, 50, y1));
                }


                var changedict = new Dictionary<int, bool>();

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X >= ret[idx].X - wdr) && (item.X <= ret[idx].X + wdr))
                        {
                            var currentrect = new OpenCvSharp.Rect(item.X - 2, ret[idx].Y, item.Width + 4, ret[idx].Height);
                            if (idx == 0)
                            {
                                currentrect = new OpenCvSharp.Rect((item.X - 6) > 0 ? (item.X - 6) : 0, ret[idx].Y, item.Width + 2, ret[idx].Height);
                            }

                            if (idx == 7)
                            {
                                currentrect = new OpenCvSharp.Rect(item.X, ret[idx].Y, (edged.Width - item.Width - 4) > 0 ? (item.Width + 4) : (edged.Width - item.X - 1), ret[idx].Height);
                            }

                            //ret[idx] = currentrect;

                            if (!changedict.ContainsKey(idx) && currentrect.Width < 58)
                            {
                                ret[idx] = currentrect;
                                changedict.Add(idx, true);
                            }
                            break;
                        }//end if
                    }//end foreach
                }//end for

                for (var idx = 0; idx < 7; idx++)
                {
                    foreach (var item in filteredbond)
                    {
                        if ((item.X >= ret[idx].X - wdr) && (item.X <= ret[idx].X + wdr))
                        {
                            if ((idx >= 0 && idx <= 2) || (idx >= 4 && idx <= 6))
                            {
                                var nextrect = new OpenCvSharp.Rect(item.X + item.Width + 2, ret[idx].Y, item.Width + 4, ret[idx].Height);
                                if (idx + 1 == 7)
                                {
                                    nextrect = new OpenCvSharp.Rect(item.X + item.Width + 4, ret[idx].Y, (edged.Width - item.X - 2 * item.Width - 8) > 0 ? (item.Width + 4) : (edged.Width - item.X - item.Width - 4), ret[idx].Height);
                                }

                                if (!changedict.ContainsKey(idx + 1) && nextrect.Width < 58)
                                {
                                    ret[idx + 1] = nextrect;
                                    changedict.Add(idx + 1, true);
                                }
                            }

                            if ((idx >= 1 && idx <= 3) || (idx >= 5 && idx <= 7))
                            {
                                var nextrect = new OpenCvSharp.Rect((item.X - item.Width - 2) > 0 ? (item.X - item.Width - 2) : 0, ret[idx].Y, item.Width + 4, ret[idx].Height);
                                if (idx - 1 == 0)
                                {
                                    nextrect = new OpenCvSharp.Rect((item.X - item.Width - 6) > 0 ? (item.X - item.Width - 6) : 0, ret[idx].Y, item.Width, ret[idx].Height);
                                }

                                if (!changedict.ContainsKey(idx - 1) && nextrect.Width < 58)
                                {
                                    ret[idx - 1] = nextrect;
                                    changedict.Add(idx - 1, true);
                                }
                            }
                            break;
                        }//end if
                    }//end foreach
                }//end for
            }
            else
            {
                var y0 = 32;
                var y1 = edged.Height - 60;

                if ((int)xcmax - 226 > 0)
                {
                    ret.Add(new OpenCvSharp.Rect((int)xcmax - 226, y0, 48, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect(0, y0, 48, y1));
                }

                ret.Add(new OpenCvSharp.Rect((int)xcmax - 164, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 110, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)xcmax - 55, y0, 48, y1));

                ret.Add(new OpenCvSharp.Rect((int)ycmin - 2, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 53, y0, 48, y1));
                ret.Add(new OpenCvSharp.Rect((int)ycmin + 110, y0, 48, y1));

                if (((int)ycmin + 211) >= (edged.Cols - 1))
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, edged.Cols - (int)ycmin - 161, y1));
                }
                else
                {
                    ret.Add(new OpenCvSharp.Rect((int)ycmin + 161, y0, 50, y1));
                }

            }

            return ret;
        }


        public static List<OpenCvSharp.Rect> GetCoordBond1(Mat blurred, Mat edged,Mat xyenhance4)
        {
            var rectlist = new List<OpenCvSharp.Rect>();

            var xlow = (int)(edged.Width * 0.333 - 20);
            var xhigh = (int)(edged.Width * 0.666 - 10);

            var edged23 = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged23, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 23, 20);

            //var edged11 = new Mat(blurred.Size(),MatType.CV_8UC1);
            //Cv2.Threshold(blurred, edged11, 123, 255, ThresholdTypes.Binary);
            //Cv2.BitwiseNot(edged11, edged11);

            var struc = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(3, 3));
            var erodemat = new Mat();
            Cv2.Erode(edged, erodemat, struc);

            var matlist = new List<Mat>();
            matlist.Add(edged);
            matlist.Add(edged23);
            matlist.Add(erodemat);
            //matlist.Add(edged11);

            //using (new Window("edged23", edged23))
            //{
            //    Cv2.WaitKey();
            //}

            //using (new Window("erodemat", erodemat))
            //{
            //    Cv2.WaitKey();
            //}

            //using (new Window("edged11", edged11))
            //{
            //    Cv2.WaitKey();
            //}

            foreach (var m in matlist)
            {
                var outmat = new Mat();
                var ids = OutputArray.Create(outmat);
                var cons = new Mat[] { };
                Cv2.FindContours(m, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                var idx1 = 0;
                foreach (var item in cons)
                {
                    idx1++;

                    var crect = Cv2.BoundingRect(item);

                    if (crect.Width >= 40 && crect.Width <= 60 && crect.Height > 50 && crect.Height < 90)
                    {
                        if ((crect.X < xlow || crect.X > xhigh) && crect.X > 5 && crect.Y >= 6)
                        {
                            //Cv2.Rectangle(xyenhance4, crect, new Scalar(255, 0, 0));
                            //using (new Window("xyenhance4tb" + idx1, xyenhance4))
                            //{
                            //    Cv2.WaitKey();
                            //}

                            rectlist.Add(crect);
                        }

                    }//end if
                }
            }//end foreach

            return rectlist;
        }

        public static List<OpenCvSharp.Rect> GetCoordBond2(Mat blurred, Mat edged,Mat xyenhance4)
        {
            var rectlist = new List<OpenCvSharp.Rect>();

            var xlow = (int)(edged.Width * 0.333 - 20);
            var xhigh = (int)(edged.Width * 0.666 - 10);

            var edged23 = new Mat();
            Cv2.AdaptiveThreshold(blurred, edged23, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 23, 20);

            //var edged11 = new Mat(blurred.Size(),MatType.CV_8UC1);
            //Cv2.Threshold(blurred, edged11, 123, 255, ThresholdTypes.Binary);
            //Cv2.BitwiseNot(edged11, edged11);

            var struc = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(3, 3));
            var erodemat = new Mat();
            Cv2.Erode(edged, erodemat, struc);

            var matlist = new List<Mat>();
            matlist.Add(edged);
            matlist.Add(edged23);
            matlist.Add(erodemat);
            //matlist.Add(edged11);

            //using (new Window("edged23", edged23))
            //{
            //    Cv2.WaitKey();
            //}

            //using (new Window("erodemat", erodemat))
            //{
            //    Cv2.WaitKey();
            //}

            //using (new Window("edged11", edged11))
            //{
            //    Cv2.WaitKey();
            //}

            foreach (var m in matlist)
            {
                var outmat = new Mat();
                var ids = OutputArray.Create(outmat);
                var cons = new Mat[] { };
                Cv2.FindContours(m, out cons, ids, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                var idx1 = 0;
                foreach (var item in cons)
                {
                    idx1++;

                    var crect = Cv2.BoundingRect(item);

                    if (crect.Width >= 60 && crect.Width <= 90 && crect.Height > 75 && crect.Height < 135)
                    {
                        if ((crect.X < xlow || crect.X > xhigh) && crect.X > 5 && crect.Y >= 6)
                        {
                            Cv2.Rectangle(xyenhance4, crect, new Scalar(255, 0, 0));
                            using (new Window("xyenhance4tb" + idx1, xyenhance4))
                            {
                                Cv2.WaitKey();
                            }

                            rectlist.Add(crect);
                        }

                    }//end if
                }
            }//end foreach

            return rectlist;
        }


        private static List<double> GetCoordWidthPT1(Mat mat,Mat edged)
        {
            //var denoisemat = new Mat();
            //Cv2.FastNlMeansDenoisingColored(mat, denoisemat, 10, 10, 7, 21);

            var ret = new List<List<double>>();
            var kaze = KAZE.Create();
            var kazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null;
            kaze.DetectAndCompute(edged, null, out kazeKeyPoints, kazeDescriptors);

            var hl = 0.25 * mat.Height;
            var hh = 0.75 * mat.Height;
            var wl = 10;
            var wh = mat.Width - 10;
            var hptlist = new List<KeyPoint>();
            foreach (var pt in kazeKeyPoints)
            {
                if (pt.Pt.Y >= hl && pt.Pt.Y <= hh
                    && pt.Pt.X >= wl && pt.Pt.X <= wh)
                {
                    hptlist.Add(pt);
                }
            }

            //var wptlist = hptlist;

            var wptlist = new List<KeyPoint>();
            for (var idx = 15; idx < mat.Width;)
            {
                var ylist = new List<double>();

                var wlist = new List<KeyPoint>();
                foreach (var pt in hptlist)
                {
                    if (pt.Pt.X >= (idx - 15) && pt.Pt.X < idx)
                    {
                        wlist.Add(pt);
                        ylist.Add(pt.Pt.Y);
                    }
                }

                if (wlist.Count > 8 && (ylist.Max() - ylist.Min()) > 0.25 * mat.Height)
                { wptlist.AddRange(wlist); }
                idx = idx + 15;
            }

            var xlist = new List<double>();
            if (wptlist.Count() == 0)
            {
                return xlist;
            }

            foreach (var pt in wptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            var xlength = xlist.Max() - xlist.Min();
            var coordlength = 0.336 * xlength;
            var xmin = xlist.Min() + coordlength;
            var xmax = xlist.Max() - coordlength;

            var xyptlist = new List<KeyPoint>();
            foreach (var pt in wptlist)
            {
                if (pt.Pt.X <= xmin || pt.Pt.X >= xmax)
                {
                    xyptlist.Add(pt);
                }
            }

            var dstKaze = new Mat();
            Cv2.DrawKeypoints(mat, xyptlist.ToArray(), dstKaze);

            using (new Window("dstKaze", dstKaze))
            {
                Cv2.WaitKey();
            }

            xlist.Clear();
            foreach (var pt in xyptlist)
            {
                xlist.Add(pt.Pt.X);
            }

            return xlist;
        }


       
    
        /// <summary>
        /// Submatrix operations
        /// </summary>
        private void SubMat()
        {
            Mat src = Cv2.ImRead(FilePath.Image.ogp,ImreadModes.Color);
            var part = src.SubMat(145,180,200,420);
            Cv2.Resize(part, part, new Size(750,125));

            Mat detailEnhance = new Mat();
            Cv2.DetailEnhance(part, detailEnhance);

            //// Assign small image to mat
            //Mat small = new Mat();
            //Cv2.Resize(src, small, new Size(100, 100));
            //src[10, 110, 10, 110] = small;
            //src[370, 470, 400, 500] = small.T();
            //// ↑ This is same as the following:
            ////small.T().CopyTo(src[370, 470, 400, 500]);

            //// Get partial mat (similar to cvSetImageROI)
            //Mat part = src[200, 400, 200, 360];
            //// Invert partial pixel values
            //Cv2.BitwiseNot(part, part);

            //// Fill the region (50..100, 100..150) with color (128, 0, 0)
            //part = src.SubMat(50, 100, 400, 450);
            //part.SetTo(128);
            var kazecutpic = new Mat();
            KeyPoint[] kazecut = null;
            var akazealg = AKAZE.Create();
            akazealg.DetectAndCompute(detailEnhance, null, out kazecut, kazecutpic);

            var xlist = new List<float>();
            var ylist = new List<float>();
            foreach (var kp in kazecut)
            {
                xlist.Add(kp.Pt.X);
                ylist.Add(kp.Pt.Y);
            }

            var detailEnhance1 = detailEnhance.SubMat(Convert.ToInt32(ylist.Min()-5), Convert.ToInt32(ylist.Max()+8), Convert.ToInt32(xlist.Min()-5), Convert.ToInt32(xlist.Max()+5));

            var kaze = KAZE.Create();
            var akaze = AKAZE.Create();

            var kazeDescriptors = new Mat();
            var akazeDescriptors = new Mat();
            KeyPoint[] kazeKeyPoints = null, akazeKeyPoints = null;
            
                kaze.DetectAndCompute(detailEnhance, null, out kazeKeyPoints, kazeDescriptors);
                akaze.DetectAndCompute(detailEnhance, null, out akazeKeyPoints, akazeDescriptors);

            var dstKaze = new Mat();
            var dstAkaze = new Mat();
            Cv2.DrawKeypoints(detailEnhance, kazeKeyPoints, dstKaze);
            Cv2.DrawKeypoints(detailEnhance, akazeKeyPoints, dstAkaze);

            BRISK brisk = BRISK.Create();
            KeyPoint[] brkkeypoints = brisk.Detect(detailEnhance);
            var briskmk = new Mat();
            Cv2.DrawKeypoints(detailEnhance, brkkeypoints, briskmk);

            //var gray = new Mat();
            //Cv2.CvtColor(detailEnhance, gray, ColorConversionCodes.BGR2GRAY);



            using (new Window("briskmk", briskmk))
            using (new Window("KAZE", dstKaze))
            using (new Window("AKAZE", dstAkaze))
            using (new Window("src", src))
            using (new Window("SubMat", detailEnhance))
            using (new Window("detailEnhance1", detailEnhance1))
            {
                Cv2.WaitKey();
            }
        }

        /// <summary>
        /// Submatrix operations
        /// </summary>
        private void RowColRangeOperation()
        {
            Mat src = Cv2.ImRead(FilePath.Image.Lenna);

            Cv2.GaussianBlur(
                src.RowRange(100, 200),
                src.RowRange(200, 300),
                new Size(7, 7), 20);

            Cv2.GaussianBlur(
                src.ColRange(200, 300),
                src.ColRange(100, 200),
                new Size(7, 7), 20);

            using (new Window("RowColRangeOperation", src))
            {
                Cv2.WaitKey();
            }
        }

        /// <summary>
        /// Submatrix expression operations
        /// </summary>
        //private void RowColOperation()
        //{
        //    Mat src = Cv2.ImRead(FilePath.Image.Lenna);

        //    Random rand = new Random();
        //    for (int i = 0; i < 200; i++)
        //    {
        //        int c1 = rand.Next(100, 400);
        //        int c2 = rand.Next(100, 400);
        //        Mat temp = src.Row[c1];
        //        src.Row[c1] = src.Row[c2];
        //        src.Row[c2] = temp;
        //    }

        //    src.Col[0, 50] = ~src.Col[450, 500];
            
        //    // set constant value (not recommended)
        //    src.Row[450,460] = src.Row[450,460] * 0 + new Scalar(0,0,255);
        //    // recommended way
        //    //src.RowRange(450, 460).SetTo(new Scalar(0, 0, 255));

        //    using (new Window("RowColOperation", src))
        //    {
        //        Cv2.WaitKey();
        //    }
        //}

    }
}