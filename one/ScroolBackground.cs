using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScroolBackground : MonoBehaviour
{
    public float speed = -2f;
    public float lowerY = -20f;
    public float upperY = 40;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.Translate(0, speed * Time.deltaTime, 0);
        if (transform.position.y <= lowerY)
        {
            transform.Translate(0, upperY, 0);
        }
    }
}
