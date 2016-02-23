using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_classify
{
    class TestDocs
    {
        public Dictionary<string, bool> FeatureDict;
        public string correctClass;
        public string curClass;
        public List<TBL_train.Transformation> TransformList;

        public TestDocs (string CC, string predClass)
        {
            FeatureDict = new Dictionary<string, bool>();
            correctClass = CC;
            curClass = predClass;
            TransformList = new List<TBL_train.Transformation>();
        }
    }
}
