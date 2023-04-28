      public void GetXYOBJ(string path, string xyfile, string outpath)
        {
            var files = System.IO.Directory.EnumerateFiles(path).ToList();
            var sb = new StringBuilder();
            foreach (var f in files)
            {
                if (f.ToUpper().Contains(".JPG"))
                {
                    //var objlist = Find5X1XYOBJ(f, 25, 43, 4800, 8000, outpath);
                    //var objlist = FindSIXINCHOBJ( f, outpath);
                    //var objlist = FindF2X1XYOBJ(f, outpath);
                    //var objlist = FindA10OBJ(f, outpath);
                    var objlist = FindIIVIOBJ(f, outpath);
                    foreach (var obj in objlist)
                    {
                        if (obj.claid != 0 && obj.iw != 0)
                        {
                            sb.Append(obj.ih);
                            sb.Append(";");
                            sb.Append(obj.iw);
                            sb.Append(";");
                            sb.Append(obj.xmin);
                            sb.Append(";");
                            sb.Append(obj.xmax);
                            sb.Append(";");
                            sb.Append(obj.ymin);
                            sb.Append(";");
                            sb.Append(obj.ymax);
                            sb.Append(";");
                            sb.Append(obj.fpath);
                            sb.Append(";");
                            sb.Append(obj.clatx);
                            sb.Append(";");
                            sb.Append(obj.claid);
                            sb.Append("\n");
                        }
                    }
                }
            }

            if (sb.Length > 10)
            { File.WriteAllText(xyfile, sb.ToString());}
            
        }
        