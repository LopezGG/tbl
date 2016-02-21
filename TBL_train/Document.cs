using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_train
{
    class Document
    {

        public Dictionary<string, bool> FeatureDict;
        public string correctClass;
        public string curClass;

        public Document (string CC,string predClass)
        {
            FeatureDict = new Dictionary<string, bool>();
            correctClass = CC;
            curClass = predClass;
        }
    }
}
