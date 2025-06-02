using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Metoer : MonoBehaviour
{
    public float speed = -2f;
    Rigidbody2D rigidbody;
    // Start is called before the first frame update
    void Start()
    {
        rigidbody  =GetComponent<Rigidbody2D>();
        rigidbody.velocity = new Vector2(0, speed);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
