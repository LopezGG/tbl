using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TBL_classify
{
    class Program
    {
        static void Main (string[] args)
        {
            string TestFile = args[0];
            string modelFile = args[1];
            string SysOutput = args[2];
            int N = Convert.ToInt32(args[3]);

            
            //read the ModelFIle and store transformations
            List<TBL_train.Transformation> TransList = ReadModel(modelFile,N);

            //get TestData in place
            Dictionary<String, List<int>> FeatureDocDict = new Dictionary<string, List<int>>();
            Dictionary<int, TestDocs> testDocDict = new Dictionary<int, TestDocs>();
            List<string> UniqueClasses = new List<string>();
            List<String> Vocab = ReadTestFile(TestFile, ref FeatureDocDict, ref testDocDict,ref UniqueClasses);
            string feature="",fromClass="",toClass="";
            for (int i = 0; i < TransList.Count; i++)
            {
                List<int> DocIds;
                feature= TransList[i].TriggerFeature;
                if (FeatureDocDict.ContainsKey(feature))
                    DocIds = FeatureDocDict[feature].ToList();
                else
                    continue;
                fromClass = TransList[i].FromClass;
                toClass = TransList[i].ToClass;
                foreach (var docId in DocIds)
                {
                    if(testDocDict.ContainsKey(docId) && testDocDict[docId].curClass == fromClass)
                    {
                        testDocDict[docId].curClass = toClass;
                        testDocDict[docId].TransformList.Add(TransList[i]);
                    }
                } 
            }
            testDocDict = testDocDict.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value);
            Dictionary<string, int> ConfusionDict = new Dictionary<string, int>();
            string key;
            using (StreamWriter Sw = new StreamWriter(SysOutput))
            {
                foreach (var Doc in testDocDict)
                {
                    Sw.Write("File: " + Convert.ToString(Doc.Key) + " " + Doc.Value.correctClass + " " + Doc.Value.curClass+" ");
                    key = Doc.Value.correctClass + "_" + Doc.Value.curClass;
                    if (ConfusionDict.ContainsKey(key))
                        ConfusionDict[key]++;
                    else
                        ConfusionDict.Add(key, 1);
                    for (int i = 0; i < Doc.Value.TransformList.Count; i++)
                    {
                        Sw.Write(Doc.Value.TransformList[i].TriggerFeature + " " + Doc.Value.TransformList[i].FromClass + "  " + Doc.Value.TransformList[i].ToClass);
                    }
                    Sw.WriteLine();
                }
            }
            
            WriteConfusionMatrix(UniqueClasses, ConfusionDict, "test", testDocDict.Keys.Count);
            Console.ReadLine();
        }
        public static void WriteConfusionMatrix (List<String> ClassBreakDown, Dictionary<String, int> ConfusionDict, string testOrTrain, int totalInstances)
        {
            int correctPred = 0;
            Console.WriteLine("Confusion matrix for the " + testOrTrain + " data:\n row is the truth, column is the system output");
            Console.Write("\t\t\t");
            foreach (var actClass in ClassBreakDown)
            {
                Console.Write(actClass + "\t");
            }
            Console.WriteLine();
            foreach (var actClass in ClassBreakDown)
            {

                Console.Write(actClass + "\t");
                foreach (var predClass in ClassBreakDown)
                {

                    if (ConfusionDict.ContainsKey(actClass + "_" + predClass))
                    {
                        Console.Write(ConfusionDict[actClass + "_" + predClass] + "\t");
                        if (actClass == predClass)
                            correctPred += ConfusionDict[actClass + "_" + predClass];
                    }
                    else
                        Console.Write("0" + "\t");

                }
                Console.WriteLine();
            }
            Console.WriteLine(testOrTrain + " accuracy=" + Convert.ToString(correctPred / ( double )totalInstances));
            Console.WriteLine();


        }
        public static List<String> ReadTestFile (string TestFile, ref  Dictionary<String, List<int>> FeatureDocDict, ref Dictionary<int, TestDocs> testDocDict, ref List<string> UniqueClasses)
        {
            List<string> Vocab = new List<string>();
            string line,firstClass="",key;
            bool flag = true;
            int docId = 0;
            using (StreamReader Sr = new StreamReader(TestFile))
            {
                while ((line = Sr.ReadLine()) != null)
                {
                    if (String.IsNullOrWhiteSpace(line))
                        continue;

                    string[] words = line.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                    if (flag)
                    {
                        firstClass = words[0];
                        flag = false;
                    }
                    UniqueClasses.Add(words[0]);
                    TestDocs temp = new TestDocs(words[0], firstClass);
                    for (int i = 1; i < words.Length; i++)
                    {
                        key = words[i].Substring(0, words[i].IndexOf(":"));
                        if (!temp.FeatureDict.ContainsKey(key))
                            temp.FeatureDict.Add(key, true);

                        if (!FeatureDocDict.ContainsKey(key))
                            FeatureDocDict.Add(key, new List<int>() { docId });
                        else
                            FeatureDocDict[key].Add(docId);
                    }
                    testDocDict.Add(docId, temp);
                    docId++;
                }
                Vocab = FeatureDocDict.Keys.ToList();
            }
            UniqueClasses = UniqueClasses.Distinct().ToList();
            return Vocab;
        }
        public static List<TBL_train.Transformation> ReadModel (string modelFile,int N)
        {
            string line;
            List<TBL_train.Transformation> TransList = new List<TBL_train.Transformation>();
            int transCount = 1;
            using (StreamReader Sr = new StreamReader(modelFile))
            {
                while ((line = Sr.ReadLine()) != null)
                {
                    if (transCount > N)
                        break;
                    if (String.IsNullOrWhiteSpace(line))
                        continue;
                    
                    string[] words = line.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length < 4)
                        throw new Exception("Invalid Model Format");
                    TransList.Add(new TBL_train.Transformation(words[0], words[1], words[2], Convert.ToInt32(words[3])));
                    transCount++;
                }
            }
            return TransList;
        }
    }
}
