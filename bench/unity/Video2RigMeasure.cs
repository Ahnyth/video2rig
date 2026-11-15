// Bench only: how Unity's animation import changes a video2rig motion. The same FBX imported several ways (Assets/Raw
// copies set here, Assets/Video2Rig/creature.fbx by Video2RigImport.cs), every clip sampled at 60 Hz, every bone's world
// position written to measure.json for bench/unity_compare.py.
//     Unity -batchmode -nographics -projectPath <project> -executeMethod Video2RigMeasure.Run -quit
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class Video2RigMeasure
{
    struct Variant
    {
        public string Path, Label;
        public ModelImporterAnimationCompression? Compression;
        public float Error;
    }

    static readonly Variant[] Variants =
    {
        new Variant { Path = "Assets/Raw/creature_off.fbx", Label = "off", Compression = ModelImporterAnimationCompression.Off },
        new Variant { Path = "Assets/Raw/creature_default.fbx", Label = "unity default (Optimal 0.5)",
                      Compression = ModelImporterAnimationCompression.Optimal, Error = .5f },
        new Variant { Path = "Assets/Raw/creature_reduction1.fbx", Label = "keyframe reduction 1.0",
                      Compression = ModelImporterAnimationCompression.KeyframeReduction, Error = 1f },
        new Variant { Path = "Assets/Video2Rig/creature.fbx", Label = "Video2RigImport.cs" },
    };

    static string F(float v) => v.ToString("0.#####", CultureInfo.InvariantCulture);

    public static void Run()
    {
        foreach (var v in Variants.Where(v => v.Compression.HasValue))
        {
            var importer = (ModelImporter)AssetImporter.GetAtPath(v.Path);
            importer.animationType = ModelImporterAnimationType.Generic;
            importer.animationCompression = v.Compression.Value;
            importer.animationRotationError = importer.animationPositionError = importer.animationScaleError = v.Error;
            importer.SaveAndReimport();
        }
        var sb = new StringBuilder("{\"variants\": [");
        for (int vi = 0; vi < Variants.Length; vi++)
        {
            var v = Variants[vi];
            var importer = (ModelImporter)AssetImporter.GetAtPath(v.Path);
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(v.Path);
            var go = Object.Instantiate(model);
            var bones = go.GetComponentsInChildren<Transform>().Where(t => t != go.transform).ToArray();
            var clips = AssetDatabase.LoadAllAssetsAtPath(v.Path).OfType<AnimationClip>()
                                     .Where(c => !c.name.StartsWith("__preview__")).OrderBy(c => c.name).ToArray();
            sb.Append(vi > 0 ? ", " : "").Append("{\"label\": \"").Append(v.Label).Append("\", \"root_node\": \"")
              .Append(importer.motionNodeName).Append("\", \"compression\": \"").Append(importer.animationCompression)
              .Append("\", \"bones\": [").Append(string.Join(", ", bones.Select(b => "\"" + b.name + "\""))).Append("], \"clips\": [");
            for (int ci = 0; ci < clips.Length; ci++)
            {
                var clip = clips[ci];
                var setting = importer.clipAnimations.FirstOrDefault(c => c.name == clip.name);
                sb.Append(ci > 0 ? ", " : "").Append("{\"name\": \"").Append(clip.name).Append("\", \"length\": ")
                  .Append(F(clip.length)).Append(", \"frame_rate\": ").Append(F(clip.frameRate))
                  .Append(", \"loop_time\": ").Append(setting != null && setting.loopTime ? "true" : "false")
                  .Append(", \"bytes\": ").Append(UnityEngine.Profiling.Profiler.GetRuntimeMemorySizeLong(clip))
                  .Append(", \"frames\": [");
                int n = Mathf.FloorToInt(clip.length * 60f + 1e-3f) + 1;
                for (int i = 0; i < n; i++)
                {
                    clip.SampleAnimation(go, Mathf.Min(i / 60f, clip.length));
                    sb.Append(i > 0 ? ", " : "").Append("[")
                      .Append(string.Join(", ", bones.Select(b => F(b.position.x) + ", " + F(b.position.y) + ", " + F(b.position.z))))
                      .Append("]");
                }
                sb.Append("]}");
            }
            sb.Append("]}");
            Object.DestroyImmediate(go);
        }
        sb.Append("]}");
        File.WriteAllText("measure.json", sb.ToString());
        Debug.Log("VIDEO2RIG_MEASURE written measure.json");
    }
}
