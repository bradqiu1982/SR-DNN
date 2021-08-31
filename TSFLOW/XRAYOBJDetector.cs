using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Web;

namespace SkyEye.Models
{
    public class XRAYItem
    {
        public static List<XRAYItem> Parse(string json)
        {
            return (List<XRAYItem>)Newtonsoft.Json.JsonConvert.DeserializeObject(json, (new List<XRAYItem>()).GetType());
        }

        public XRAYItem() {

            level = "";
            rate = "";
            image = "";
        }
        public string level { set; get; }
        public string rate { set; get; }
        public string image { set; get; }
        public string imgpath { get { return @"file:///" + image.Replace("\\", "/"); } }
    }

    public class XRAYOBJDetector
    {
        public static List<XRAYItem> XRAYOBJDect(string imgpath)
        {
            var ret = new List<XRAYItem>();
            var pathobj = new
            {
                imgpath = imgpath
            };

            var reqstr = Newtonsoft.Json.JsonConvert.SerializeObject(pathobj);
            var response = PythonRESTFun("http://wux-engsys01.chn.ii-vi.net:5000/XRAYOBJDetect", reqstr);
            if (!string.IsNullOrEmpty(response))
            {
                ret = XRAYItem.Parse(response);
            }
            return ret;
        }

        private static string PythonRESTFun(string url, string reqstr)
        {
            string webResponse = string.Empty;
            try
            {
                Uri uri = new Uri(url);
                WebRequest httpWebRequest = (HttpWebRequest)WebRequest.Create(uri);
                httpWebRequest.ContentType = "application/json";
                httpWebRequest.Method = "POST";
                httpWebRequest.Timeout = -1;
                ((HttpWebRequest)httpWebRequest).ReadWriteTimeout = -1;

                using (StreamWriter streamWriter = new StreamWriter(httpWebRequest.GetRequestStream()))
                {
                    streamWriter.Write(reqstr);
                }

                HttpWebResponse httpWebResponse = (HttpWebResponse)httpWebRequest.GetResponse();
                if (httpWebResponse.StatusCode == HttpStatusCode.OK)
                {
                    using (StreamReader streamReader = new StreamReader(httpWebResponse.GetResponseStream()))
                    {
                        webResponse = streamReader.ReadToEnd();
                    }
                }
            }
            catch (Exception ex)
            {
            }

            return webResponse;
        }
    }
}