using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MetoerSpawn : MonoBehaviour
{
    public GameObject metoer;
    public float minSpawn = 1;
    public float maxSpawn = 3;
    public float spawnXlimit = 6;
    // Start is called before the first frame update
    void Start()
    {
        Spawn();
    }

    // Update is called once per frame
    void Update()
    {
        
 
    }
    public void Spawn()
    {
        float random = Random.Range(-spawnXlimit, spawnXlimit);
        Vector3 spwanPos = transform.position + new Vector3(random, 0, 0);
        Instantiate(metoer, spwanPos, Quaternion.identity);
        Invoke("Spawn",Random.Range(minSpawn, maxSpawn));

    }
}
