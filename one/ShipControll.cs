using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShipControll : MonoBehaviour
{
    private GameManager gameManager;
  
    public GameObject bullet;
    public float speed = 10f;
    public float xlimit=7f;
    public float reloadTime = 0.5f;
    float elapsedTime = 0;


    // Start is called before the first frame update
    void Start()
    {
        gameManager = FindAnyObjectByType<GameManager>();
    }

    // Update is called once per frame
    void Update()
    {
        elapsedTime += Time.deltaTime;
        float xInput = Input.GetAxis("Horizontal");
        float yInput = Input.GetAxis("Vertical");
        transform.Translate(xInput * speed * Time.deltaTime, yInput*Time.deltaTime*speed, 0);
        Vector3 pos = transform.position;
        pos.x = Mathf.Clamp(pos.x, -xlimit, xlimit);
        pos.y = Mathf.Clamp(pos.y,-xlimit, xlimit);
        transform.position = pos;
        if (Input.GetButtonDown("Jump") && elapsedTime > reloadTime)
        {
            gameManager.PlayNhacChieu();
            Vector3 spwanPos = transform.position;
                
            spwanPos += new Vector3(0, 1.2f, 0);
            Instantiate(bullet, spwanPos, Quaternion.identity);
            elapsedTime = 0;
            

        }

    }
    private void OnTriggerEnter2D(Collider2D collision)
    {
        gameManager.PlayerDied();
    }
}
