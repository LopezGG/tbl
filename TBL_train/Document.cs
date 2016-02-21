using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_train
{
    class Document
    {
        public int DocId;
        public Dictionary<string, bool> FeatureDict;
        public string correctClass;
        public string curClass;

        public Document (int DiD, string CC,string predClass)
        {
            DocId = DiD;
            FeatureDict = new Dictionary<string, bool>();
            correctClass = CC;
            curClass = predClass;
        }
    }
}
