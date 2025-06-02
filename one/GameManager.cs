using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class GameManager : MonoBehaviour
{
    public Text scoreText;
    public Text gameOverText;
    int playerScore = 0;
    public AudioSource NhacNenn;
    public AudioSource NhacChieu;
    public AudioClip NhacChieuClip;
    public AudioClip NhacNenClip;
    // Start is called before the first frame update
    void Start()
    {
        NhacNenn.clip = NhacNenClip;
        NhacNenn.Play();
    }
    public void PlayNhacChieu()
    {
        NhacChieu.PlayOneShot(NhacChieuClip);
    }
    // Update is called once per frame
    void Update()
    {
        
    }

    public void Addscore()
    {
        playerScore++;
        scoreText.text = playerScore.ToString();
    }

    public void PlayerDied()
    {
        gameOverText.enabled = true;
        Time.timeScale = 0;
    }
}
